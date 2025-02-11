import json
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM

def load_mtop_data(mtop_sampled_json):
    """Loads the MTop sampled JSON data."""
    with open(mtop_sampled_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data  

def load_pairs(pairs_json):
    """Loads the contrastive pairs from the new JSON file."""
    with open(pairs_json, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    return pairs

def get_llama_hidden_states_batched(model, tokenizer, texts, device, target_layer=None, batch_size=8):
    """
    Given a list of texts, compute the LLaMA hidden representations in batches.
    The output is the mean-pooled hidden representation from the target layer (or last layer if None).
    """
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=True
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
    all_hs = outputs.hidden_states
    if target_layer is None:
        chosen_hs = all_hs[-1]
    else:
        chosen_hs = all_hs[target_layer]
    # Mean pool the hidden states over the sequence length (using attention mask)
    mask_float = attention_mask.unsqueeze(-1).expand(chosen_hs.size()).float()
    sum_hs = (chosen_hs * mask_float).sum(dim=1)
    lengths = attention_mask.sum(dim=1).unsqueeze(-1)
    mean_hs = sum_hs / lengths
    return mean_hs.cpu().numpy()

def main():
    # File paths (update these as needed)
    mtop_sampled_json = "../data/mtop/en/mtop_train_sampled.json"   # MTop sampled data
    pairs_json = "../scripts/mtop_contrastive_pairs_with_embedding_filter.json"    # New contrastive pairs JSON file
    output_npy = "llama2_reps_infonce_with_embedding_filter.npz"                                # Output NPZ file
    target_layer = None  # Use last layer if None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8

    print("Loading MTop dataset...")
    data = load_mtop_data(mtop_sampled_json)
    print("Loading contrastive pairs...")
    pairs = load_pairs(pairs_json)

    # Initialize LLaMA‑2‑7B model and tokenizer
    model_name = "meta-llama/Llama-2-7b-hf"
    print(f"Loading LLaMA2-7B from {model_name} ...")
    access_token = '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'
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()

    # Gather all unique indices from the pairs:
    # - anchors: p["anchor_idx"]
    # - positives: p["positive"]["candidate_idx"]
    # - negatives: p["negatives"] (a list)
    anchor_idxs = [p["anchor_idx"] for p in pairs]
    positive_idxs = [p["positive"]["candidate_idx"] for p in pairs]
    negatives_idxs = []
    for p in pairs:
        negatives_idxs.extend(p["negatives"])
    unique_indices = set(anchor_idxs + positive_idxs + negatives_idxs)
    unique_indices_list = sorted(list(unique_indices))
    print(f"Total unique sentences needed: {len(unique_indices_list)}")

    # Create a mapping from index to original sentence text (used for extracting hidden representations)
    idx_to_text = {}
    for idx in unique_indices_list:
        idx_to_text[idx] = data[idx]["original_sentence"]

    # Batch-process all unique sentences to extract LLaMA hidden representations.
    index_to_rep = {}
    all_texts = [idx_to_text[idx] for idx in unique_indices_list]
    all_reps = []
    print("Extracting LLaMA hidden representations...")
    for start in tqdm(range(0, len(all_texts), batch_size), desc="Extracting LLaMA reps"):
        batch_texts = all_texts[start:start+batch_size]
        reps = get_llama_hidden_states_batched(model, tokenizer, batch_texts, device, target_layer=target_layer, batch_size=batch_size)
        for rep in reps:
            all_reps.append(rep)
    # Map each unique index to its hidden representation
    for i, idx in enumerate(unique_indices_list):
        index_to_rep[idx] = all_reps[i]

    # Construct final arrays for each contrastive pair.
    # For each pair, we want to save:
    #   - anchor representation (vector)
    #   - positive representation (vector)
    #   - negatives representations (array of shape (n_negatives, hidden_dim))
    #   - parse_similarity (float, from the positive block)
    #   - (optionally) the indices for reference.
    anchor_reps = []
    positive_reps = []
    negatives_reps = []  # List of arrays; lengths may vary per sample.
    parse_sims = []
    anchor_indices = []
    positive_indices = []
    negatives_indices = []  # List of lists

    print(f"Constructing final arrays for {len(pairs)} pairs...")
    for p in tqdm(pairs, desc="Constructing arrays"):
        a_idx = p["anchor_idx"]
        pos_idx = p["positive"]["candidate_idx"]
        neg_idxs = p["negatives"]  # list of negative indices
        psim = p["positive"].get("parse_similarity", np.nan)
  
        a_rep = index_to_rep[a_idx]
        pos_rep = index_to_rep[pos_idx]
        neg_reps = np.stack([index_to_rep[n] for n in neg_idxs], axis=0)
        
        anchor_reps.append(a_rep)
        positive_reps.append(pos_rep)
        negatives_reps.append(neg_reps)
        parse_sims.append(psim)
        anchor_indices.append(a_idx)
        positive_indices.append(pos_idx)
        negatives_indices.append(neg_idxs)
    
    anchor_arr = np.stack(anchor_reps, axis=0)       # shape: (N_pairs, hidden_dim)
    positive_arr = np.stack(positive_reps, axis=0)     # shape: (N_pairs, hidden_dim)
    parse_sim_arr = np.array(parse_sims, dtype=float)  # shape: (N_pairs,)
    anchor_indices_arr = np.array(anchor_indices)
    positive_indices_arr = np.array(positive_indices)
    # Negatives are of variable length per sample; store as an object array.
    negatives_arr = np.array(negatives_reps, dtype=object)
    negatives_indices_arr = np.array(negatives_indices, dtype=object)
    
    print(f"anchor_arr shape: {anchor_arr.shape}")
    print(f"positive_arr shape: {positive_arr.shape}")
    print(f"negatives_arr length: {len(negatives_arr)}")
    print(f"parse_sim_arr shape: {parse_sim_arr.shape}")
    
    print(f"Saving to {output_npy} ...")
    np.savez(
        output_npy,
        anchor=anchor_arr,
        positive=positive_arr,
        negatives=negatives_arr,
        parse_similarity=parse_sim_arr,
        anchor_idx=anchor_indices_arr,
        positive_idx=positive_indices_arr,
        negatives_idx=negatives_indices_arr
    )
    print("Done.")

if __name__ == "__main__":
    main()
