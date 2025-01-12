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

    with open(mtop_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # data: list of dict, each => {"original_sentence":..., "mtop_parsing":...}
    N = len(data)
    print(f"Loaded {N} MTop samples from {mtop_json_path}.")

    model_name = "all-MiniLM-L6-v2"  
    embedder = SentenceTransformer(model_name)
    utterances = [item["original_sentence"] for item in data]

    print(f"Embedding {N} sentences with {model_name}...")
    embeddings = embedder.encode(utterances, batch_size=32, show_progress_bar=True)
    embeddings = embeddings.astype(np.float32)

    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d) 
    index.add(embeddings)
    print("Faiss index built.")

    final_pairs = []
    D, I = index.search(embeddings, top_k+1)

    all_indices = set(range(N))  
    for i in range(N):
        neighbors = I[i].tolist()
        if i in neighbors:
            neighbors.remove(i)
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


        outside_candidates = list(all_indices - {i} - set(neighbors))
        if len(outside_candidates) >= random_neg_outside:
            neg_samples = random.sample(outside_candidates, random_neg_outside)
            for neg_j in neg_samples:
                final_pairs.append({
                    "anchor_idx": i,
                    "candidate_idx": neg_j,
                    "parse_similarity": None,  
                    "label": "neg"
                })
        else:
            pass


    with open(output_pairs_path, "w", encoding="utf-8") as fw:
        json.dump(final_pairs, fw, ensure_ascii=False, indent=2)

    print(f"Collected {len(final_pairs)} pairs => saved to {output_pairs_path}.")

  
    positive_count = sum(1 for pair in final_pairs if pair["label"] == "pos")
    negative_count = sum(1 for pair in final_pairs if pair["label"] == "neg")

    print(f"Number of positive pairs: {positive_count}")
    print(f"Number of negative pairs: {negative_count}")



if __name__ == "__main__":

    mtop_json = "../data/mtop/en/mtop_train.json"
    output_pairs = "mtop_contrastive_pairs.json"

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