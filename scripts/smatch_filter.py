
'''import os
import json
import math
import random
import concurrent.futures

import smatch
from smatch import get_amr_match, compute_f

def compute_smatch_f1(amr_str1, amr_str2):
    """
    计算两棵 AMR 的 Smatch F1 分数 (用 smatch 的 get_amr_match + compute_f).
    """
    best_match_num, test_triple_num, gold_triple_num = get_amr_match(amr_str1, amr_str2)
    precision, recall, f1 = compute_f(best_match_num, test_triple_num, gold_triple_num)
    return f1

def process_chunk(chunk, amr_graphs, pos_thr, neg_thr):
    """
    对 chunk 中的若干对 (i, j) 做 Smatch 计算，返回符合 (f1 >= pos_thr or f1 <= neg_thr) 的结果.
    """
    results = []
    for pair in chunk:
        i = pair["anchor_idx"]
        j = pair["candidate_idx"]
        amr_str_i = amr_graphs[i]
        amr_str_j = amr_graphs[j]

        f1 = compute_smatch_f1(amr_str_i, amr_str_j)

        # 根据阈值判定最终标签
        if f1 >= pos_thr:
            new_label = "pos"
        elif f1 <= neg_thr:
            new_label = "neg"
        else:
            # 跳过中间区域
            continue

        results.append({
            "anchor_idx": i,
            "candidate_idx": j,
            "smatch_f1": f1,
            "final_label": new_label
        })
    return results

if __name__ == "__main__":
    # -------------------- 1. 读取 AMR & candidate_pairs --------------------
    with open("../data/amrs/split/training/all_amrs.json", "r", encoding="utf-8") as f:
        all_amrs = json.load(f)
    amr_graphs = [amr["graph"] for amr in all_amrs]
    print(f"Loaded {len(amr_graphs)} AMRs from all_amrs.json.")

    with open("candidate_pairs.json", "r", encoding="utf-8") as f:
        candidate_pairs = json.load(f)
    print(f"Loaded {len(candidate_pairs)} candidate pairs.")

    # -------------------- 2. 拆分 pos/neg 并随机采样 --------------------
    # 2.1 分离
    pos_pairs = []
    neg_pairs = []
    for pair in candidate_pairs:
        if pair["label"] == "pos":
            pos_pairs.append(pair)
        elif pair["label"] == "neg":
            neg_pairs.append(pair)
        # 若你还有 "cos_sim" 之类，可以保留也可忽略

    # 2.2 决定采样规模
    #    例如你想总共训练对 = 30000, 那么想要 pos:neg ~= 1:1
    max_total = 30000
    # 先算能否各取一半
    half = max_total // 2  # 15000
    # 实际可取数
    pos_needed = min(int(half*4.5), len(pos_pairs))
    neg_needed = min(half, len(neg_pairs))

    # 2.3 随机抽取
    random.shuffle(pos_pairs)
    random.shuffle(neg_pairs)
    pos_sample = pos_pairs[:pos_needed]
    neg_sample = neg_pairs[:neg_needed]

    # 2.4 合并
    sampled_pairs = pos_sample + neg_sample
    random.shuffle(sampled_pairs)
    print(f"After sampling: pos={len(pos_sample)}, neg={len(neg_sample)}, total={len(sampled_pairs)}")

    # -------------------- 3. 分块 (chunk) 并行计算 Smatch --------------------
    pos_thr = 0.6
    neg_thr = 0.2

    chunk_size = 200
    chunks = []
    for i in range(0, len(sampled_pairs), chunk_size):
        chunk = sampled_pairs[i : i + chunk_size]
        chunks.append(chunk)
    print(f"Total chunks: {len(chunks)} with chunk_size={chunk_size}")

    final_pairs = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_chunk, c, amr_graphs, pos_thr, neg_thr) for c in chunks]

        for fut in concurrent.futures.as_completed(futures):
            chunk_result = fut.result()
            final_pairs.extend(chunk_result)

    # -------------------- 4. 统计并输出结果 --------------------
    total_processed = len(sampled_pairs)  # 经过 smatch 的
    accepted_count = len(final_pairs)
    skip_count = total_processed - accepted_count

    pos_count = sum(1 for p in final_pairs if p["final_label"] == "pos")
    neg_count = sum(1 for p in final_pairs if p["final_label"] == "neg")

    # 保存
    with open("final_smatch_pairs.json", "w", encoding="utf-8") as fw:
        json.dump(final_pairs, fw, ensure_ascii=False, indent=2)

    print("===== Final Stats =====")
    print(f"Sampled Pairs for Smatch: {total_processed}")
    print(f"Skipped (middle region) : {skip_count}")
    print(f"Accepted (pos/neg)     : {accepted_count}")
    print(f"  - POS: {pos_count}")
    print(f"  - NEG: {neg_count}")'''


import os
import json
import random
import zlib
import concurrent.futures
import ijson
from smatch import get_amr_match, compute_f

def compress_graph(graph):
    return zlib.compress(graph.encode('utf-8'))

def decompress_graph(compressed_graph):
    return zlib.decompress(compressed_graph).decode('utf-8')

def compute_smatch_f1(amr_str1, amr_str2):
    """
    计算两棵 AMR 的 Smatch F1 分数。
    """
    best_match_num, test_triple_num, gold_triple_num = get_amr_match(amr_str1, amr_str2)
    precision, recall, f1 = compute_f(best_match_num, test_triple_num, gold_triple_num)
    return f1

def process_chunk(chunk, amr_graphs, pos_thr, neg_thr):
    """
    对 chunk 中的若干对 (i, j) 做 Smatch 计算，返回符合 (f1 >= pos_thr or f1 <= neg_thr) 的结果。
    """
    results = []
    for pair in chunk:
        i = pair["anchor_idx"]
        j = pair["candidate_idx"]
        amr_str_i = decompress_graph(amr_graphs[i])
        amr_str_j = decompress_graph(amr_graphs[j])

        f1 = compute_smatch_f1(amr_str_i, amr_str_j)

        if f1 >= pos_thr:
            new_label = "pos"
        elif f1 <= neg_thr:
            new_label = "neg"
        else:
            continue

        results.append({
            "anchor_idx": i,
            "candidate_idx": j,
            "smatch_f1": f1,
            "final_label": new_label
        })
    return results

def load_amrs(filepath):
    """
    使用生成器逐行加载 AMR 数据。
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        for amr in ijson.items(f, 'item'):
            yield amr

def chunkify(lst, chunk_size):
    """
    将列表分块。
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

if __name__ == "__main__":
    # -------------------- 1. 读取 AMR & candidate_pairs --------------------
    print("Loading AMRs...")
    amr_generator = load_amrs("../data/amrs/split/training/all_amrs.json")
    amr_graphs = [compress_graph(amr["graph"]) for amr in amr_generator]
    print(f"Loaded {len(amr_graphs)} AMRs.")

    print("Loading candidate pairs...")
    with open("candidate_pairs.json", "r", encoding="utf-8") as f:
        candidate_pairs = json.load(f)
    print(f"Loaded {len(candidate_pairs)} candidate pairs.")

    # -------------------- 2. 随机采样 --------------------
    pos_pairs = [pair for pair in candidate_pairs if pair["label"] == "pos"]
    neg_pairs = [pair for pair in candidate_pairs if pair["label"] == "neg"]

    max_total = 30000
    half = max_total // 2
    pos_needed = min(half*5, len(pos_pairs))
    neg_needed = min(half, len(neg_pairs))

    random.shuffle(pos_pairs)
    random.shuffle(neg_pairs)
    pos_sample = pos_pairs[:pos_needed]
    neg_sample = neg_pairs[:neg_needed]

    sampled_pairs = pos_sample + neg_sample
    random.shuffle(sampled_pairs)
    print(f"After sampling: pos={len(pos_sample)}, neg={len(neg_sample)}, total={len(sampled_pairs)}")

    # -------------------- 3. 分块并行处理 --------------------
    pos_thr = 0.6
    neg_thr = 0.2
    chunk_size = 200

    chunks = list(chunkify(sampled_pairs, chunk_size))
    print(f"Total chunks: {len(chunks)} with chunk_size={chunk_size}")

    with open("final_smatch_pairs.json", "w", encoding="utf-8") as fw:
        fw.write("[")  # 开始写入 JSON 数组
        first = True

        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_chunk, c, amr_graphs, pos_thr, neg_thr) for c in chunks]

            for fut in concurrent.futures.as_completed(futures):
                chunk_result = fut.result()
                for result in chunk_result:
                    if not first:
                        fw.write(",\n")
                    json.dump(result, fw, ensure_ascii=False)
                    first = False

        fw.write("]")  # 结束 JSON 数组

    # -------------------- 4. 打印统计信息 --------------------
    print("===== Final Stats =====")
    print(f"Sampled Pairs for Smatch: {len(sampled_pairs)}")
    print(f"Output saved to final_smatch_pairs.json.")

