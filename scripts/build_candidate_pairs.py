import json
import random

def build_candidate_pairs(all_amrs, I, D, pos_k=5, neg_k=5):

    n = len(all_amrs)
    candidate_pairs = []

    all_indices = list(range(n))

    for i in range(n):
        pos_candidates = I[i][1:pos_k+1]  
        for c_idx in pos_candidates:
            pair_info = {
                "anchor_idx": int(i),  
                "candidate_idx": int(c_idx), 
                "label": "pos",
                "cos_sim": float(D[i][list(I[i]).index(c_idx)]) 
            }
            candidate_pairs.append(pair_info)

      
        candidate_pool = [idx for idx in all_indices if idx != i]
        neg_list = random.sample(candidate_pool, neg_k)

        for c_idx in neg_list:
            pair_info = {
                "anchor_idx": int(i),
                "candidate_idx": int(c_idx),
                "label": "neg",
                "cos_sim": None 
            }
            candidate_pairs.append(pair_info)

    return candidate_pairs



if __name__ == "__main__":
    import numpy as np

    with open("../data/amrs/split/training/all_amrs.json", "r", encoding="utf-8") as f:
        all_amrs = json.load(f)
    N = len(all_amrs)
    print("Total AMRs:", N)

    I = np.load("I.npy")
    D = np.load("D.npy")

    pos_k = 5
    neg_k = 5
    candidate_pairs = build_candidate_pairs(all_amrs, I, D, pos_k, neg_k)
    print(f"Total candidate pairs: {len(candidate_pairs)}")

    out_file = "candidate_pairs.json"
    with open(out_file, "w", encoding="utf-8") as fw:
        json.dump(candidate_pairs, fw, ensure_ascii=False, indent=2)
    print(f"Candidate pairs saved to {out_file}")
