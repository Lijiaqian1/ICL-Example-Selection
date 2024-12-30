'''import json
import penman
import smatch

def compute_smatch_f1(amr_str1, amr_str2):
    """
    计算两棵 AMR 的 Smatch F1。
    如果传入的是 penman.Graph 对象，请先在外部 penman.encode() 成字符串。
    """
    best_match_num, test_match_num, gold_match_num = smatch.get_amr_match(amr_str1, amr_str2)
    precision = best_match_num / test_match_num if test_match_num else 0
    recall    = best_match_num / gold_match_num if gold_match_num else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

if __name__ == "__main__":

    # 1. 读取所有 AMR 数据（包含 'graph' 信息）
    with open("../data/amrs/split/training/all_amrs.json", "r", encoding="utf-8") as f:
        all_amrs = json.load(f)
    N = len(all_amrs)
    print(f"Loaded {N} AMRs from all_amrs.json.")

    # 2. 读取候选对 (candidate_pairs) 
    with open("candidate_pairs.json", "r", encoding="utf-8") as f:
        candidate_pairs = json.load(f)
    print(f"Loaded {len(candidate_pairs)} candidate pairs.")

    # 3. 设定 Smatch 筛选阈值
    pos_thr = 0.7
    neg_thr = 0.3

    # 4. 遍历每个候选对，计算 Smatch，并决定是否纳入最终的正负例
    final_pairs = []
    skip_count = 0

    for pair in candidate_pairs:
        i = pair["anchor_idx"]
        j = pair["candidate_idx"]
        orig_label = pair["label"]  # "pos" or "neg" (粗检索时标的)

        # 读取 AMR graph
        # 如果 all_amrs[i]["graph"] 存的是 penman.Graph 对象，你需要先做 penman.encode
        # 如果已经是字符串，则可直接用。
        amr_str_i = all_amrs[i]["graph"]
        amr_str_j = all_amrs[j]["graph"]

        # 若不是字符串，可手动 encode:
        # amr_str_i = penman.encode(all_amrs[i]["graph"])
        # amr_str_j = penman.encode(all_amrs[j]["graph"])

        # 计算 Smatch F1
        f1 = compute_smatch_f1(amr_str_i, amr_str_j)

        # 根据 F1 得分决定新的 label
        if f1 >= pos_thr:
            new_label = "pos"
        elif f1 <= neg_thr:
            new_label = "neg"
        else:
            # 如果处于中间区域，不纳入最终对（可以根据需要自行处理）
            skip_count += 1
            continue

        final_pairs.append({
            "anchor_idx": i,
            "candidate_idx": j,
            "smatch_f1": f1,
            "final_label": new_label
        })

    # 5. 将最终对比学习对保存到 JSON 文件
    out_file = "final_smatch_pairs.json"
    with open(out_file, "w", encoding="utf-8") as fw:
        json.dump(final_pairs, fw, ensure_ascii=False, indent=2)
    print(f"Saved {len(final_pairs)} final pairs to {out_file}")

    # 6. 统计 & 打印结果
    total_count = len(final_pairs)
    pos_count = sum(1 for p in final_pairs if p["final_label"] == "pos")
    neg_count = sum(1 for p in final_pairs if p["final_label"] == "neg")

    print("====== Final Stats ======")
    print(f"Skipped middle region pairs (f1 in ({neg_thr}, {pos_thr})): {skip_count}")
    print(f"Total accepted pairs: {total_count}")
    print(f"  - POS pairs: {pos_count}")
    print(f"  - NEG pairs: {neg_count}")'''




import concurrent.futures
import json
import smatch

def compute_smatch_f1(amr_str1, amr_str2):
    # 1. 解析和对比 AMR 图
    best_match_num, test_triple_num, gold_triple_num = get_amr_match(amr_str1, amr_str2)
    
    # 2. 计算 F1 分数
    precision, recall, f1 = compute_f(best_match_num, test_triple_num, gold_triple_num)
    
    return f1


def process_pair(pair, amr_graphs, pos_thr, neg_thr):
    """
    处理单个候选对，计算 Smatch 并根据阈值判断是否纳入最终对。
    """
    i = pair["anchor_idx"]
    j = pair["candidate_idx"]

    amr_str_i = amr_graphs[i]
    amr_str_j = amr_graphs[j]

    f1 = compute_smatch_f1(amr_str_i, amr_str_j)

    if f1 >= pos_thr:
        new_label = "pos"
    elif f1 <= neg_thr:
        new_label = "neg"
    else:
        return None  # 跳过中间区域对

    return {
        "anchor_idx": i,
        "candidate_idx": j,
        "smatch_f1": f1,
        "final_label": new_label
    }

if __name__ == "__main__":
    # 1. 读取所有 AMR 数据（包含 'graph' 信息）和候选对 (candidate_pairs)
    with open("../data/amrs/split/training/all_amrs.json", "r", encoding="utf-8") as f:
        all_amrs = json.load(f)

    with open("candidate_pairs.json", "r", encoding="utf-8") as f:
        candidate_pairs = json.load(f)

    # 提取 graph 数据（优化访问速度）
    amr_graphs = [amr["graph"] for amr in all_amrs]

    # 设置 Smatch 阈值
    pos_thr = 0.7
    neg_thr = 0.3

    # 2. 并行处理
    final_pairs = []
    skip_count = 0

    # 使用 ProcessPoolExecutor 并行处理候选对
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # 提交任务
        futures = [
            executor.submit(process_pair, pair, amr_graphs, pos_thr, neg_thr)
            for pair in candidate_pairs
        ]

        # 收集结果
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                final_pairs.append(result)
            else:
                skip_count += 1

    # 3. 保存最终的对
    with open("final_smatch_pairs.json", "w", encoding="utf-8") as fw:
        json.dump(final_pairs, fw, ensure_ascii=False, indent=2)

    # 4. 打印统计信息
    total_count = len(final_pairs)
    pos_count = sum(1 for p in final_pairs if p["final_label"] == "pos")
    neg_count = sum(1 for p in final_pairs if p["final_label"] == "neg")

    print("====== Final Stats ======")
    print(f"Skipped middle region pairs (f1 in ({neg_thr}, {pos_thr})): {skip_count}")
    print(f"Total accepted pairs: {total_count}")
    print(f"  - POS pairs: {pos_count}")
    print(f"  - NEG pairs: {neg_count}")
