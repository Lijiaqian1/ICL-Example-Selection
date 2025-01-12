# gather_contrastive_pairs_smcalflow.py

'''import json
import random
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from smcalflow_parse_sim import lispress_tree_similarity

def gather_contrastive_pairs_smcalflow(
    smcalflow_json_path,
    output_pairs_path="smcalflow_contrastive_pairs.json",
    top_k=5,
    parse_thr=0.4,
    random_neg_outside=2
):
    

    with open(smcalflow_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)


    N = len(data)
    print(f"Loaded {N} SMCalFlow samples from {smcalflow_json_path}.")

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

        anchor_parse = data[i]["parse"]
        for j in neighbors:
            cand_parse = data[j]["parse"]
            sim = lispress_tree_similarity(anchor_parse, cand_parse)
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


    with open(output_pairs_path, "w", encoding="utf-8") as fw:
        json.dump(final_pairs, fw, ensure_ascii=False, indent=2)

    print(f"Collected {len(final_pairs)} pairs => saved to {output_pairs_path}.")

    positive_count = sum(1 for pair in final_pairs if pair["label"] == "pos")
    negative_count = sum(1 for pair in final_pairs if pair["label"] == "neg")

    print(f"Number of positive pairs: {positive_count}")
    print(f"Number of negative pairs: {negative_count}")



if __name__ == "__main__":

    smcalflow_json = "./smcalflow_preprocessed.json"
    output_pairs = "smcalflow_contrastive_pairs.json"


    TOP_K = 5
    PARSE_THR = 0.4
    RAND_NEG_OUTSIDE = 2

    gather_contrastive_pairs_smcalflow(
        smcalflow_json_path=smcalflow_json,
        output_pairs_path=output_pairs,
        top_k=TOP_K,
        parse_thr=PARSE_THR,
        random_neg_outside=RAND_NEG_OUTSIDE
    )'''


import json
import random
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from concurrent.futures import ProcessPoolExecutor, as_completed
from smcalflow_parse_sim import lispress_tree_similarity

def _process_chunk(task_batch, data, parse_thr):
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
            sim = lispress_tree_similarity(parseA, parseB)
            label = "pos" if sim >= parse_thr else "neg"
            results.append({
                "anchor_idx": ai,
                "candidate_idx": aj,
                "parse_similarity": float(sim),
                "label": label
            })
    return results

def gather_contrastive_pairs_smcalflow_parallel(
    smcalflow_json_path,
    output_pairs_path="smcalflow_contrastive_pairs.json",
    top_k=5,
    parse_thr=0.5,
    random_neg_outside=0,
    chunk_size=200,
    max_workers=4
):
    with open(smcalflow_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    N = len(data)
    print(f"Loaded {N} SMCalFlow samples from {smcalflow_json_path}.")

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

    # chunk
    chunked_tasks = []
    for start in range(0, len(candidate_pairs), chunk_size):
        chunk = candidate_pairs[start:start+chunk_size]
        chunked_tasks.append(chunk)

    final_pairs = []

    print(f"Start parallel parse-sim calc: #chunks={len(chunked_tasks)}, chunk_size={chunk_size}, max_workers={max_workers}")

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
    pos_count = sum(1 for x in final_pairs if x["label"] == "pos")
    neg_count = sum(1 for x in final_pairs if x["label"] == "neg")
    print(f"POS: {pos_count}, NEG: {neg_count}")

if __name__ == "__main__":
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    smcalflow_json = "./smcalflow_train_sampled.json"
    output_pairs = "./smcalflow_contrastive_pairs.json"

    gather_contrastive_pairs_smcalflow_parallel(
        smcalflow_json_path=smcalflow_json,
        output_pairs_path=output_pairs,
        top_k=5,
        parse_thr=0.4,
        random_neg_outside=0,
        chunk_size=200,
        max_workers=4
    )

