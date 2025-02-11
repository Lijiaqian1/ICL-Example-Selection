import json
import random
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from concurrent.futures import ProcessPoolExecutor, as_completed
from mtop_parse_sim import mtop_tree_similarity


def _process_chunk(task_batch, data, parse_thr, k1_neg, k2_neg):
    """
    处理子进程中的 MTop parsing 相似度计算任务。
    task_batch: [(anchor_idx, top_k_candidates)], 其中：
      - anchor_idx: anchor 句子索引
      - top_k_candidates: embedding 初筛出的 top-K 候选索引
    data: 包含 "mtop_parsing" 的列表
    parse_thr: float
    k1_neg: int, 从 Top-K 内部选取的负例数量
    k2_neg: int, 从 Top-K 之外选取的负例数量
    返回：[{完整的 JSON 结构}, ...]
    """
    results = []

    for anchor_idx, candidate_idxs in task_batch:
        anchor_sent = data[anchor_idx]["original_sentence"]
        anchor_parse = data[anchor_idx]["mtop_parsing"]

        # Step 1: 计算 Top-K 里所有候选的 mtop parse 相似度
        candidate_sims = []
        for cand_idx in candidate_idxs:
            cand_parse = data[cand_idx]["mtop_parsing"]
            sim = mtop_tree_similarity(anchor_parse, cand_parse)
            candidate_sims.append((cand_idx, sim))

        # Step 2: 选取 mtop parse 相似度最高的作为正例
        candidate_sims.sort(key=lambda x: x[1], reverse=True)
        pos_idx, pos_sim = candidate_sims[0]  # 最高相似度的作为正例

        # Step 3: 从 Top-K 内部随机选 k1 个负例（排除正例）
        remaining_top_k = [idx for idx, _ in candidate_sims[1:]]  # 除去正例的索引
        neg_k1 = random.sample(remaining_top_k, min(k1_neg, len(remaining_top_k)))

        # Step 4: 从 Top-K 之外随机选 k2 个负例
        all_indices = set(range(len(data)))
        remaining_outside = list(all_indices - set(candidate_idxs) - {anchor_idx})
        neg_k2 = random.sample(remaining_outside, min(k2_neg, len(remaining_outside)))

        # Step 5: 组织数据结构
        anchor_data = {
            "anchor_idx": anchor_idx,
            "anchor_sentence": anchor_sent,
            "anchor_parsing": anchor_parse,
            "positive": {
                "candidate_idx": pos_idx,
                "candidate_sentence": data[pos_idx]["original_sentence"],
                "candidate_parsing": data[pos_idx]["mtop_parsing"],
                "parse_similarity": float(pos_sim)
            },
            "negatives": neg_k1 + neg_k2  # 负例仅存索引
        }
        results.append(anchor_data)

    return results


def gather_contrastive_pairs_parallel(
    mtop_json_path,
    output_pairs_path="mtop_contrastive_pairs.json",
    top_k=10,  # Embedding 相似度筛选的 Top-K
    parse_thr=0.4,
    k1_neg=2,  # 从 Top-K 内部选的负例数
    k2_neg=3,  # 从 Top-K 之外选的负例数
    chunk_size=200,
    max_workers=4
):
    """使用 embedding 初筛后，再用 MTop 解析树相似度筛选正例/负例"""
    with open(mtop_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    N = len(data)
    print(f"Loaded {N} samples from {mtop_json_path}.")

    # 计算 embedding
    model_name = "all-MiniLM-L6-v2"
    embedder = SentenceTransformer(model_name)
    utterances = [item["original_sentence"] for item in data]
    print(f"Embedding {N} sentences with {model_name}...")
    embeddings = embedder.encode(utterances, batch_size=32, show_progress_bar=True).astype(np.float32)

    # 构建 Faiss 近邻索引
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    D, I = index.search(embeddings, top_k + 1)  # 找 top-K+1, 以排除自身
    print("Faiss index built. Searching top_k done.")

    candidate_pairs = []  # [(anchor_idx, top_k_candidates)]
    
    for i in range(N):
        neighbors = I[i].tolist()
        if i in neighbors:
            neighbors.remove(i)  # 移除自己本身
        candidate_pairs.append((i, neighbors))  # 记录 Anchor 和候选集

    print(f"Total candidate pairs: {len(candidate_pairs)}")

    # 分块处理
    chunked_tasks = [candidate_pairs[i:i+chunk_size] for i in range(0, len(candidate_pairs), chunk_size)]
    final_pairs = []

    print(f"Starting parallel parse similarity calculations with {max_workers} workers...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_chunk, batch, data, parse_thr, k1_neg, k2_neg) for batch in chunked_tasks]
        for fut in as_completed(futures):
            final_pairs.extend(fut.result())

    # 保存最终 pairs
    with open(output_pairs_path, "w", encoding="utf-8") as fw:
        json.dump(final_pairs, fw, ensure_ascii=False, indent=2)

    print(f"Saved {len(final_pairs)} contrastive pairs to {output_pairs_path}")


if __name__ == "__main__":
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    
    mtop_json = "../data/mtop/en/mtop_train_sampled.json"
    output_pairs = "mtop_contrastive_pairs_with_embedding_filter.json"

    gather_contrastive_pairs_parallel(
        mtop_json_path=mtop_json,
        output_pairs_path=output_pairs,
        top_k=10,  # 初筛 Embedding 近邻
        parse_thr=0.4,
        k1_neg=3,  # Top-K 内部负例
        k2_neg=2,  # Top-K 之外负例
        chunk_size=200,
        max_workers=4
    )
