# gather_contrastive_pairs_parallel.py

import json
import random
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from concurrent.futures import ProcessPoolExecutor, as_completed

# 从 mtop_parse_sim 里导入 parse-level 相似度函数(基于ZSS)
from mtop_parse_sim import mtop_tree_similarity

#############################################
# 顶层函数: 给子进程使用
#############################################

def _process_chunk(task_batch, data, parse_thr):
    """
    在子进程中执行 parse 相似度计算:
    task_batch: [(anchor_idx, candidate_idx, isTopK), ...]
    data: 含 "mtop_parsing" 的列表
    parse_thr: float
    返回：[{anchor_idx, candidate_idx, parse_similarity, label}, ...]
    """
    results = []
    for (ai, aj, is_topk) in task_batch:
        if not is_topk:
            # outside topK => 直接 label=neg, parse_similarity=None
            results.append({
                "anchor_idx": ai,
                "candidate_idx": aj,
                "parse_similarity": None,
                "label": "neg"
            })
        else:
            parseA = data[ai]["mtop_parsing"]
            parseB = data[aj]["mtop_parsing"]
            sim = mtop_tree_similarity(parseA, parseB)
            label = "pos" if sim >= parse_thr else "neg"
            results.append({
                "anchor_idx": ai,
                "candidate_idx": aj,
                "parse_similarity": float(sim),
                "label": label
            })
    return results


def gather_contrastive_pairs_parallel(
    mtop_json_path,
    output_pairs_path="mtop_contrastive_pairs.json",
    top_k=5,
    parse_thr=0.4,
    random_neg_outside=0,
    chunk_size=200,
    max_workers=4
):
    """
    1) 读取 MTop 数据(含 original_sentence, mtop_parsing)
    2) 对 original_sentence 做 GPU embedding, Faiss检索 topK
    3) 生成 candidate pairs
       - topK => 并行计算 parse相似度 => pos/neg
       - outside => neg
    4) 写出 final pairs
    """
    with open(mtop_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    N = len(data)
    print(f"Loaded {N} samples from {mtop_json_path}.")

    # === GPU embedding in main process ===
    model_name = "all-MiniLM-L6-v2"
    embedder = SentenceTransformer(model_name)
    utterances = [item["original_sentence"] for item in data]

    print(f"Embedding {N} sentences with {model_name} on main process (likely GPU)...")
    embeddings = embedder.encode(utterances, batch_size=32, show_progress_bar=True)
    embeddings = embeddings.astype(np.float32)

    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    D, I = index.search(embeddings, top_k+1)
    print("Faiss index built. Searching top_k done.")

    all_indices = set(range(N))

    candidate_pairs = []  # list of (i, j, isTopK)
    for i in range(N):
        neighbors = I[i].tolist()
        if i in neighbors:
            neighbors.remove(i)
        # topK
        for j in neighbors:
            candidate_pairs.append((i, j, True))
        # outside
        outside_candidates = list(all_indices - {i} - set(neighbors))
        if len(outside_candidates) >= random_neg_outside:
            neg_samples = random.sample(outside_candidates, random_neg_outside)
            for neg_j in neg_samples:
                candidate_pairs.append((i, neg_j, False))

    print(f"Total raw candidate pairs: {len(candidate_pairs)}")

    # 分 chunk
    chunked_tasks = []
    for start in range(0, len(candidate_pairs), chunk_size):
        chunk = candidate_pairs[start:start+chunk_size]
        chunked_tasks.append(chunk)

    final_pairs = []

    print(f"Start parallel parse-sim calc: #chunks={len(chunked_tasks)}, chunk_size={chunk_size}, max_workers={max_workers}")

    # === 这里 parse-level similarity 在子进程进行 ===
    #    但不会 re-init GPU, 因为 zss/embedding is typically CPU-based
    #    or if it uses GPU => we need spawn mode
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for batch in chunked_tasks:
            fut = executor.submit(_process_chunk, batch, data, parse_thr)
            futures.append(fut)
        for fut in as_completed(futures):
            batch_result = fut.result()
            final_pairs.extend(batch_result)

    with open(output_pairs_path, "w", encoding="utf-8") as fw:
        json.dump(final_pairs, fw, ensure_ascii=False, indent=2)

    print(f"Saved {len(final_pairs)} pairs to {output_pairs_path}")
    pos_count = sum(1 for x in final_pairs if x["label"]=="pos")
    neg_count = sum(1 for x in final_pairs if x["label"]=="neg")
    print(f"POS: {pos_count}, NEG: {neg_count}")


########################
# Main
########################
if __name__ == "__main__":
    # 在 Linux, 如果默认 "fork" 导致 CUDA re-init错误 => 强制 spawn
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # 可能已经设置过

    # 你可以把路径改成实际文件
    mtop_json = "../data/mtop/en/mtop_train_sampled.json"
    output_pairs = "mtop_contrastive_pairs.json"

    gather_contrastive_pairs_parallel(
        mtop_json_path=mtop_json,
        output_pairs_path=output_pairs,
        top_k=5,
        parse_thr=0.4,
        random_neg_outside=0,
        chunk_size=200,
        max_workers=4
    )