import json
import random
import re
import numpy as np
import faiss
from tqdm import tqdm  # Progress bar library
from datasketch import MinHash, MinHashLSH
from sentence_transformers import SentenceTransformer
from mtop_parse_sim import mtop_tree_similarity

# Initialize sentence embedding model
model_name = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(model_name)

def extract_labels_from_mtop(mtop_parse):
    """ Extracts a Bag of Node Labels from the MTop parsing string """
    tokens = re.findall(r'\w+:[\w_]+|\w+', mtop_parse)
    return set(tokens)

def create_minhash(label_set, num_perm=64):
    """ Generates a MinHash signature for a given set of node labels """
    m = MinHash(num_perm=num_perm)
    for label in label_set:
        m.update(label.encode('utf8'))
    return m

def build_lsh_index(data, num_perm=64, lsh_threshold=0.4):
    """ Builds an LSH index using MinHash signatures for efficient retrieval """
    print("[Step 1] Building LSH index...")
    lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)
    minhashes = {}
    
    for i, item in tqdm(enumerate(data), total=len(data), desc="Processing MinHash"):
        parse_str = item["mtop_parsing"]
        labels = extract_labels_from_mtop(parse_str)
        m = create_minhash(labels, num_perm)
        minhashes[i] = m
        lsh.insert(str(i), m)
    
    print(f"LSH index built. Processed {len(data)} samples.")
    return lsh, minhashes

def compute_sentence_embeddings(data):
    """ Computes sentence embeddings for all original sentences in the dataset """
    sentences = [item["original_sentence"] for item in data]
    embeddings = embedder.encode(sentences, batch_size=32, show_progress_bar=True)
    return embeddings.astype(np.float32)

def find_top_k_similar_sentences(embeddings, index, k=10):
    """ Finds the top-k most similar sentences using FAISS nearest neighbor search """
    d = embeddings.shape[1]
    faiss_index = index
    faiss_index.add(embeddings)
    D, I = faiss_index.search(embeddings, k + 1)  # +1 to exclude self-matching
    
    # Ensure all indices are valid
    valid_I = []
    for neighbors in I:
        valid_neighbors = [idx for idx in neighbors if 0 <= idx < len(embeddings)]  # Keep only valid indices
        valid_I.append(valid_neighbors)
    return valid_I


def gather_contrastive_pairs_with_lsh(
    mtop_json_path,
    output_pairs_path="mtop_contrastive_pairs_with_lsh.json",
    parse_thr=0.4,
    random_negatives_count=2,
    num_perm=64,
    lsh_threshold=0.0,
    max_lsh_candidates=20,  # Maximum candidates to keep from LSH results
    top_k_fallback=10
):
    print(f"Loading data from: {mtop_json_path}")
    with open(mtop_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    N = len(data)
    print(f"Loaded {N} MTop samples.")

    # Step 1: Build LSH index
    lsh, minhashes = build_lsh_index(data, num_perm=num_perm, lsh_threshold=lsh_threshold)

    # Step 2: Compute sentence embeddings for fallback retrieval
    print("[Step 2] Computing sentence embeddings for fallback retrieval...")
    embeddings = compute_sentence_embeddings(data)
    sentence_index = faiss.IndexFlatIP(embeddings.shape[1])

    final_pairs = []
    all_indices = set(range(N))
    
    # Statistics counters
    lsh_zero_count = 0
    lsh_less_than_5_count = 0
    lsh_less_than_10_count = 0

    print("[Step 3] Processing contrastive learning samples...")
    for i in tqdm(range(N), desc="Processing Anchors",mininterval = 30):
        anchor_parse = data[i]["mtop_parsing"]
        candidate_keys = lsh.query(minhashes[i])
        candidate_indices = set(int(k) for k in candidate_keys if int(k) != i)

        # Log the number of candidates found by LSH
        candidate_count = len(candidate_indices)
        #print(f"Anchor {i}: LSH initial candidates = {candidate_count}")

        # Step 3.1: Restrict the candidate set to max_lsh_candidates if too large
        if candidate_count > max_lsh_candidates:
            candidate_indices = set(list(candidate_indices)[:max_lsh_candidates])
            #print(f"Anchor {i}: LSH candidates trimmed to {max_lsh_candidates}")

        # Update statistics
        if candidate_count == 0:
            lsh_zero_count += 1
        if candidate_count < 5:
            lsh_less_than_5_count += 1
        if candidate_count < 10:
            lsh_less_than_10_count += 1

        # Step 3.2: If LSH returns no candidates, use sentence embedding similarity as fallback
        if not candidate_indices:
            #print(f"Anchor {i}: No LSH candidates found, using sentence embedding fallback.")
            top_k_neighbors = find_top_k_similar_sentences(embeddings, sentence_index, top_k_fallback)
            
            # Convert FAISS results to a valid candidate set
            candidate_indices = set(idx for idx in top_k_neighbors[i] if 0 <= idx < len(data)) - {i}  # Ensure index is within range

            if not candidate_indices:
                #print(f"[Warning] Anchor {i}: Even sentence embedding fallback returned no valid candidates.")
                continue  # Skip if no valid candidates found

        
        # Step 3.3: Find the best positive example using mtop_tree_similarity
        best_candidate = None
        best_sim = -1.0
        for j in candidate_indices:
            cand_parse = data[j]["mtop_parsing"]
            sim = mtop_tree_similarity(anchor_parse, cand_parse)
            if sim > best_sim:
                best_sim = sim
                best_candidate = j
        
        if best_candidate is None:
            #print(f"[Warning] Anchor {i}: No suitable positive example found.")
            continue
        
        positive_example = {
            "candidate_idx": best_candidate,
            "candidate_sentence": data[best_candidate]["original_sentence"],
            "candidate_parsing": data[best_candidate]["mtop_parsing"],
            "parse_similarity": float(best_sim)
        }
        
        # Step 3.4: Select negative examples
        negatives = []
        remaining_candidates = candidate_indices - {best_candidate}
        if len(remaining_candidates) >= random_negatives_count:
            neg_indices = random.sample(list(remaining_candidates), random_negatives_count)
        else:
            remaining_candidates = all_indices - {i, best_candidate}
            neg_indices = random.sample(list(remaining_candidates), min(random_negatives_count, len(remaining_candidates)))
        for neg_idx in neg_indices:
            negatives.append(neg_idx)

        final_pairs.append({
            "anchor_idx": i,
            "anchor_sentence": data[i]["original_sentence"],
            "anchor_parsing": anchor_parse,
            "positive": positive_example,
            "negatives": negatives
        })
    
    def convert_to_serializable(obj):
        """ Recursively convert numpy.int64 and other non-serializable types to JSON serializable formats """
        if isinstance(obj, np.int64):  # Convert numpy.int64 to Python int
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj

    # Step 4: Convert and Save results
    print(f"[Step 4] Processing complete. Converting data to JSON serializable format...")
    final_pairs_serializable = convert_to_serializable(final_pairs)

    with open(output_pairs_path, "w", encoding="utf-8") as fw:
        json.dump(final_pairs_serializable, fw, ensure_ascii=False, indent=2)

    print(f"Results saved to {output_pairs_path}")


    # Print statistics
    print(f"\n==== LSH Candidate Statistics ====")
    print(f"Anchors with 0 LSH candidates: {lsh_zero_count}")
    print(f"Anchors with <5 LSH candidates: {lsh_less_than_5_count}")
    print(f"Anchors with <10 LSH candidates: {lsh_less_than_10_count}")

if __name__ == "__main__":
    mtop_json = "../data/mtop/en/mtop_train_sampled.json"
    output_pairs = "mtop_contrastive_pairs_with_lsh.json"

    gather_contrastive_pairs_with_lsh(
        mtop_json_path=mtop_json,
        output_pairs_path=output_pairs,
        parse_thr=0.4,
        random_negatives_count=5
    )
