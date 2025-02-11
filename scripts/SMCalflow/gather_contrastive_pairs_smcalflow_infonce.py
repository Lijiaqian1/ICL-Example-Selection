import json
import random
import re
import numpy as np
import faiss
from concurrent.futures import ProcessPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
from datasketch import MinHash, MinHashLSH
from smcalflow_parse_sim import lispress_tree_similarity

# -------------------------
# 1) MinHash 相关函数
# -------------------------

def extract_labels_from_lispress(parse_str):
    """ 从 SMCalFlow 解析中提取 bag of node labels """
    tokens = re.findall(r'\w+:[\w_]+|\w+', parse_str)
    return set(tokens)

def create_minhash(label_set, num_perm=64):
    """ 生成 MinHash 签名 """
    m = MinHash(num_perm=num_perm)
    for label in label_set:
        m.update(label.encode('utf8'))
    return m

def build_lsh_index(data, num_perm=64, lsh_threshold=0.4):
    """ 生成 MinHash LSH 索引 """
    print("[Step 1] 建立 LSH 索引...")
    lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)
    minhashes = {}

    for i, item in enumerate(data):
        labels = extract_labels_from_lispress(item["parse"])
        m = create_minhash(labels, num_perm)
        minhashes[i] = m
        lsh.insert(str(i), m)

    print(f"LSH 索引建立完成，共处理 {len(data)} 条数据")
    return lsh, minhashes

# -------------------------
# 2) 计算对比对（避免子进程加载 `SentenceTransformer`）
# -------------------------

def _process_chunk(task_batch, data, parse_thr, embeddings_dict):
    """ 计算 lispress_tree_similarity，但不在子进程加载 `SentenceTransformer` """
    results = []
    for (ai, aj, is_topk) in task_batch:
        if not is_topk:
            results.append({
                "anchor_idx": ai,
                "candidate_idx": aj,
                "parse_similarity": None,
                "label": "neg"
            })
        else:
            parseA = data[ai]["parse"]
            parseB = data[aj]["parse"]
            # **只计算 parse 相似度，不调用 `model.encode()`**
            sim = lispress_tree_similarity(parseA, parseB, embeddings_dict)
            label = "pos" if sim >= parse_thr else "neg"
            results.append({
                "anchor_idx": ai,
                "candidate_idx": aj,
                "parse_similarity": float(sim),
                "label": label
            })
    return results

def gather_contrastive_pairs_smcalflow_lsh(
    smcalflow_json_path,
    output_pairs_path="smcalflow_contrastive_pairs.json",
    parse_thr=0.4,
    num_perm=64,
    lsh_threshold=0.4,
    max_lsh_candidates=20,
    random_neg_count=2,
    chunk_size=200,
    max_workers=4
):
    """ 使用 MinHash + LSH 进行高效相似度筛选 """
    with open(smcalflow_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    N = len(data)
    print(f"加载 {N} 条 SMCalFlow 数据")

    # Step 1: 建立 LSH 索引
    lsh, minhashes = build_lsh_index(data, num_perm=num_perm, lsh_threshold=lsh_threshold)

    all_indices = set(range(N))
    candidate_pairs = []

    # Step 2: **预计算所有句子的 embedding**（避免子进程重复调用 `model.encode()`）
    print("[Step 2] 计算所有句子的 embedding...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings_dict = {i: model.encode(data[i]["original_sentence"]) for i in range(N)}

    # Step 3: LSH 检索
    print("[Step 3] 通过 LSH 近似搜索找到相似的候选项...")
    for i in range(N):
        neighbors = set(int(k) for k in lsh.query(minhashes[i]) if int(k) != i)
        if len(neighbors) > max_lsh_candidates:
            neighbors = set(list(neighbors)[:max_lsh_candidates])

        # 统计 LSH 结果
        if len(neighbors) == 0:
            print(f"Warning: 样本 {i} 无 LSH 候选项")

        # Step 3.1: 选取最优正例
        for j in neighbors:
            candidate_pairs.append((i, j, True))  # 需要计算 parse 相似度

        # Step 3.2: 选择随机负例
        remaining_candidates = all_indices - {i} - neighbors
        if len(remaining_candidates) >= random_neg_count:
            neg_samples = random.sample(list(remaining_candidates), random_neg_count)
            for neg_j in neg_samples:
                candidate_pairs.append((i, neg_j, False))

    print(f"总共生成 {len(candidate_pairs)} 个候选对")

    # Step 4: 并行计算 lispress_tree_similarity
    chunked_tasks = [candidate_pairs[i:i+chunk_size] for i in range(0, len(candidate_pairs), chunk_size)]
    final_pairs = []

    print(f"启动并行计算 (进程数={max_workers})")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_chunk, chunk, data, parse_thr, embeddings_dict) for chunk in chunked_tasks]
        for fut in as_completed(futures):
            final_pairs.extend(fut.result())

    # Step 5: 存储结果
    with open(output_pairs_path, "w", encoding="utf-8") as fw:
        json.dump(final_pairs, fw, ensure_ascii=False, indent=2)

    print(f"最终生成 {len(final_pairs)} 对比对，存储至 {output_pairs_path}")
    pos_count = sum(1 for x in final_pairs if x["label"] == "pos")
    neg_count = sum(1 for x in final_pairs if x["label"] == "neg")
    print(f"正例: {pos_count}, 负例: {neg_count}")

if __name__ == "__main__":
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    smcalflow_json = "./smcalflow_test_sampled_100.json"
    output_pairs = "./smcalflow_contrastive_infonce_pairs.json"

    gather_contrastive_pairs_smcalflow_lsh(
        smcalflow_json_path=smcalflow_json,
        output_pairs_path=output_pairs,
        parse_thr=0.4,
        num_perm=64,
        lsh_threshold=0.4,
        max_lsh_candidates=20,
        random_neg_count=2,
        chunk_size=200,
        max_workers=4
    )
