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

    best_match_num, test_triple_num, gold_triple_num = get_amr_match(amr_str1, amr_str2)
    precision, recall, f1 = compute_f(best_match_num, test_triple_num, gold_triple_num)
    return f1

def process_chunk(chunk, amr_graphs, pos_thr, neg_thr):

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

    with open(filepath, 'r', encoding='utf-8') as f:
        for amr in ijson.items(f, 'item'):
            yield amr

def chunkify(lst, chunk_size):

    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

if __name__ == "__main__":
    print("Loading AMRs...")
    amr_generator = load_amrs("../data/amrs/split/training/all_amrs.json")
    amr_graphs = [compress_graph(amr["graph"]) for amr in amr_generator]
    print(f"Loaded {len(amr_graphs)} AMRs.")

    print("Loading candidate pairs...")
    with open("candidate_pairs.json", "r", encoding="utf-8") as f:
        candidate_pairs = json.load(f)
    print(f"Loaded {len(candidate_pairs)} candidate pairs.")

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

    pos_thr = 0.6
    neg_thr = 0.2
    chunk_size = 200

    chunks = list(chunkify(sampled_pairs, chunk_size))
    print(f"Total chunks: {len(chunks)} with chunk_size={chunk_size}")

    with open("final_smatch_pairs.json", "w", encoding="utf-8") as fw:
        fw.write("[") 
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

        fw.write("]")  
    print("===== Final Stats =====")
    print(f"Sampled Pairs for Smatch: {len(sampled_pairs)}")
    print(f"Output saved to final_smatch_pairs.json.")

