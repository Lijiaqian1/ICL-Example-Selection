# gather_contrastive_pairs.py

import json
import random
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from mtop_parse_sim import mtop_tree_similarity

def gather_contrastive_pairs(
    mtop_json_path,
    output_pairs_path="mtop_contrastive_pairs.json",
    top_k=5,
    parse_thr=0.4,
    random_neg_outside=2
):
    """
    1) 读取 mtop_train.json, each item = {"original_sentence", "mtop_parsing"}
    2) 用 SentenceTransformer 对 original_sentence 做embedding, Faiss建索引
    3) 对每个样本 i, 检索 top_k; 
       - parse相似度 >= parse_thr => label='pos'
       - parse相似度 < parse_thr => label='neg'
       - 再额外随机采样 random_neg_outside 个来自 outside topK 作为 negative
    4) 存储结果到 output_pairs_path
    """

    # ========== 1) 读取 mtop 数据 ==========
    with open(mtop_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # data: list of dict, each => {"original_sentence":..., "mtop_parsing":...}
    N = len(data)
    print(f"Loaded {N} MTop samples from {mtop_json_path}.")

    # ========== 2) embedding + Faiss ==========
    model_name = "all-MiniLM-L6-v2"  # 你也可换别的
    embedder = SentenceTransformer(model_name)
    utterances = [item["original_sentence"] for item in data]

    print(f"Embedding {N} sentences with {model_name}...")
    embeddings = embedder.encode(utterances, batch_size=32, show_progress_bar=True)
    embeddings = embeddings.astype(np.float32)

    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # 内积
    index.add(embeddings)
    print("Faiss index built.")

    # ========== 3) 收集正负例对 ==========
    final_pairs = []
    # 先为 convenience, 预先查询 topK+1 => skip self
    D, I = index.search(embeddings, top_k+1)

    all_indices = set(range(N))  # 用于随机负例

    for i in range(N):
        # topK+1结果 => I[i], 其第0项通常是 i 自身
        neighbors = I[i].tolist()
        # 移除 i 自己
        if i in neighbors:
            neighbors.remove(i)

        # topK (去掉self后只剩 topK 了)
        # parse相似度精算
        anchor_parse = data[i]["mtop_parsing"]
        for j in neighbors:
            cand_parse = data[j]["mtop_parsing"]
            sim = mtop_tree_similarity(anchor_parse, cand_parse)
            label = "pos" if sim >= parse_thr else "neg"

            final_pairs.append({
                "anchor_idx": i,
                "candidate_idx": j,
                "parse_similarity": float(sim),
                "label": label
            })

        # 另外从 outside topK 里再随机抽 random_neg_outside 个当额外负例
        # outside = all集 - {i} - neighbors
        outside_candidates = list(all_indices - {i} - set(neighbors))
        if len(outside_candidates) >= random_neg_outside:
            neg_samples = random.sample(outside_candidates, random_neg_outside)
            for neg_j in neg_samples:
                # parse不需要精算 => 直接 label=neg
                # (也可做 parse sim < parse_thr check, 但多半是远的embedding anyway)
                final_pairs.append({
                    "anchor_idx": i,
                    "candidate_idx": neg_j,
                    "parse_similarity": None,  # or do mtop_tree_similarity if you like
                    "label": "neg"
                })
        else:
            # not enough outside
            pass

    # ========== 4) 写出结果 ==========
    with open(output_pairs_path, "w", encoding="utf-8") as fw:
        json.dump(final_pairs, fw, ensure_ascii=False, indent=2)

    print(f"Collected {len(final_pairs)} pairs => saved to {output_pairs_path}.")

    # ========== 统计正负例数量 ==========
    positive_count = sum(1 for pair in final_pairs if pair["label"] == "pos")
    negative_count = sum(1 for pair in final_pairs if pair["label"] == "neg")

    print(f"Number of positive pairs: {positive_count}")
    print(f"Number of negative pairs: {negative_count}")


# ========== DEMO 主函数 ==========

if __name__ == "__main__":
    # 例示: 
    # 假设 mtop_train.json 存在, e.g. "mtop_train.json"
    # 里面若干 {"original_sentence":..., "mtop_parsing":...}
    mtop_json = "../data/mtop/en/mtop_train.json"
    output_pairs = "mtop_contrastive_pairs.json"

    # 你可以自由设置
    TOP_K = 5
    PARSE_THR = 0.4
    RAND_NEG_OUTSIDE = 2

    gather_contrastive_pairs(
        mtop_json_path=mtop_json,
        output_pairs_path=output_pairs,
        top_k=TOP_K,
        parse_thr=PARSE_THR,
        random_neg_outside=RAND_NEG_OUTSIDE
    )