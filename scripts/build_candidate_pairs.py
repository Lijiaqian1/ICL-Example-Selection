import json
import random

def build_candidate_pairs(all_amrs, I, D, pos_k=5, neg_k=5):
    """
    基于 Faiss 检索结果，构造候选对 (i, j):
      - 正例(Top-K 最相似)  => label="pos" (后续你可根据实际需要改 label=1)
      - 负例(最不相似 or 随机抽取) => label="neg" (后续可改 label=0)

    参数说明：
    - all_amrs: 你的 AMR 数据列表，长度为 N
    - I: faiss.search 返回的下标数组, 形状 (N, K)
         I[i][0] 通常就是 i 自身，后面是最相似的其他下标
    - D: faiss.search 返回的相似度数组, 形状 (N, K)
         D[i][k] 是 i 与 I[i][k] 的内积或余弦相似度
    - pos_k: 为每个 i 选多少个正例候选
    - neg_k: 为每个 i 选多少个负例候选
    
    返回：candidate_pairs 列表，里面是 dict，形如：
    {
      "anchor_idx": i,
      "candidate_idx": j,
      "label": "pos" or "neg",
      "cos_sim": ...  # 仅供参考
    }
    """


    n = len(all_amrs)
    candidate_pairs = []

    # 预先构造一个完整的 index 列表，便于随机负例抽取
    all_indices = list(range(n))

    for i in range(n):
        # ------ 正例候选 ------
        pos_candidates = I[i][1:pos_k+1]  # 取最相似的 pos_k 个
        for c_idx in pos_candidates:
            pair_info = {
                "anchor_idx": int(i),  # 转换为 Python 的 int
                "candidate_idx": int(c_idx),  # 转换为 Python 的 int
                "label": "pos",
                "cos_sim": float(D[i][list(I[i]).index(c_idx)])  # 转换为 float
            }
            candidate_pairs.append(pair_info)

        # ------ 负例候选 ------
        candidate_pool = [idx for idx in all_indices if idx != i]
        neg_list = random.sample(candidate_pool, neg_k)

        for c_idx in neg_list:
            pair_info = {
                "anchor_idx": int(i),
                "candidate_idx": int(c_idx),
                "label": "neg",
                "cos_sim": None  # 负例相似度为 None
            }
            candidate_pairs.append(pair_info)

    return candidate_pairs



if __name__ == "__main__":
    import numpy as np

    # ---- 假设你在某处已经得到了 all_amrs, I, D ----
    # 演示时我们用模拟数据，这里请改成你自己的加载方式

    # 1. 加载 all_amrs
    with open("../data/amrs/split/training/all_amrs.json", "r", encoding="utf-8") as f:
        all_amrs = json.load(f)
    N = len(all_amrs)
    print("Total AMRs:", N)

    # 2. 加载 I, D (Faiss 检索结果)
    # 在真实情况下，你会在同一个脚本或之前的脚本中做:
    # D, I = index.search(embeddings, K)
    # 这里演示，假设你已经存了 I, D 到 .npy 文件:
    #   np.save("I.npy", I)
    #   np.save("D.npy", D)
    # 现在只需加载即可
    I = np.load("I.npy")
    D = np.load("D.npy")

    # 3. 生成候选对
    pos_k = 5
    neg_k = 5
    candidate_pairs = build_candidate_pairs(all_amrs, I, D, pos_k, neg_k)
    print(f"Total candidate pairs: {len(candidate_pairs)}")

    # 4. 将 candidate pairs 写入 JSON 以便后续使用
    out_file = "candidate_pairs.json"
    with open(out_file, "w", encoding="utf-8") as fw:
        json.dump(candidate_pairs, fw, ensure_ascii=False, indent=2)
    print(f"Candidate pairs saved to {out_file}")
