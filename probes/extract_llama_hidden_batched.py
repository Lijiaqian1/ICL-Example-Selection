# extract_llama_hidden_batched.py

import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM

def load_mtop_data(mtop_sampled_json):
    """
    Load the dataset entries => a list of dict with 'original_sentence', ...
    Return them as a list for indexing.
    """
    with open(mtop_sampled_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data  # e.g. data[i] => {"original_sentence": ..., "mtop_parsing": ...}

def load_pairs(pairs_json):
    """
    Load the pairs => a list of dict with 
       'anchor_idx', 'candidate_idx', 'label', 'parse_similarity' ...
    """
    with open(pairs_json, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    return pairs

def get_llama_hidden_states_batched(
    model,
    tokenizer,
    texts,            # list of strings
    device,
    target_layer=None
):
    """
    Batched version of hidden-state extraction for LLaMA.
    1) tokenize the entire batch
    2) run forward with output_hidden_states=True
    3) pick the layer or last layer
    4) do average pooling across seq dimension
    Return shape => [batch_size, hidden_dim]
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
    all_hs = outputs.hidden_states  # tuple (layer0,...,layerN)

    if target_layer is None:
        chosen_hs = all_hs[-1]  # (batch, seq_len, hidden_dim)
    else:
        chosen_hs = all_hs[target_layer]

    # average pooling
    # shape => (batch, hidden_dim)
    # we do a mask-based average to handle padding
    # or do a simple mean across all tokens ignoring pad
    # for simplicity, we do a naive approach: 
    # sum embeddings across seq & divide by seq_len
    # a better approach: sum only non-pad tokens
    # e.g:
    mask_float = attention_mask.unsqueeze(-1).expand(chosen_hs.size()).float()
    # chosen_hs shape => (batch, seq_len, hidden_dim)
    sum_hs = (chosen_hs * mask_float).sum(dim=1)
    lengths = mask_float.sum(dim=1)  # shape (batch, hidden_dim)
    # lengths => shape (batch, hidden_dim?), actually we did expand. let's do ...
    lengths_1d = attention_mask.sum(dim=1).unsqueeze(-1)  # shape (batch,1)
    mean_hs = sum_hs / lengths_1d  # broadcast => (batch, hidden_dim)

    return mean_hs.cpu().numpy()

def main():
    # 1) config
    mtop_sampled_json = "../data/mtop/en/mtop_train_sampled.json"      # each item => original_sentence, ...
    pairs_json = "../scripts/mtop_contrastive_pairs.json"                     # each => anchor_idx, candidate_idx, label, ...
    output_npy = "llama2_reps.npz"
    target_layer = None  # None => last layer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8  # you can adjust to smaller/larger

    # 2) load data & pairs
    print("Loading dataset entries...")
    data = load_mtop_data(mtop_sampled_json)  
    # data[i] => {"original_sentence":..., "mtop_parsing":...}
    print("Loading pairs (pos/neg + parse_similarity)...")
    pairs = load_pairs(pairs_json)
    # pairs => [ {"anchor_idx":..., "candidate_idx":..., "parse_similarity":..., "label":...}, ...]

    # 3) load LLaMA2 model
    model_name = "meta-llama/Llama-2-7b-hf"
    print(f"Loading LLaMA2-7B from {model_name} ...")
    access_token = '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'
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    # 8-bit or 4-bit if possible => but watch out for OOM
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
        # load_in_8bit=True  # optional
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()

    # 4) gather all unique sentence indices
    anchor_idxs = [p["anchor_idx"] for p in pairs]
    candidate_idxs = [p["candidate_idx"] for p in pairs]
    unique_indices = set(anchor_idxs + candidate_idxs)
    unique_indices_list = sorted(list(unique_indices))

    # 5) batch compute reps for each unique index
    index_to_rep = {}
    texts = []
    idx_map = []
    for idx_ in unique_indices_list:
        text_ = data[idx_]["original_sentence"]
        texts.append(text_)
        idx_map.append(idx_)

    print(f"Total unique sentences: {len(unique_indices_list)}")
    all_reps = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start+batch_size]
        reps = get_llama_hidden_states_batched(
            model, tokenizer, batch_texts, device, target_layer=target_layer
        )  # shape (batch_size, hidden_dim)
        for i, rid in enumerate(reps):
            all_reps.append(rid)

    # now fill index_to_rep
    # all_reps[i] corresponds to idx_map[i]
    for i, idx_ in enumerate(idx_map):
        index_to_rep[idx_] = all_reps[i]

    # 6) now build final arrays for anchor, candidate, labels, parse_sims
    anchor_reps = []
    cand_reps   = []
    labels      = []
    parse_sims  = []

    print(f"Constructing final arrays for {len(pairs)} pairs...")
    for p in tqdm(pairs):
        ai = p["anchor_idx"]
        aj = p["candidate_idx"]
        lab= p["label"]
        psim = p.get("parse_similarity", None)
        # gather anchor, cand rep from dictionary
        a_vec = index_to_rep[ai]
        c_vec = index_to_rep[aj]

        anchor_reps.append(a_vec)
        cand_reps.append(c_vec)
        labels.append(lab)
        # store parse sim => might be float or None
        parse_sims.append(psim if psim is not None else np.nan)

    anchor_arr = np.stack(anchor_reps, axis=0)  # shape [num_pairs, hidden_dim]
    cand_arr   = np.stack(cand_reps, axis=0)
    label_arr  = np.array(labels, dtype=object)
    parse_sim_arr = np.array(parse_sims, dtype=float)

    print(f"anchor_arr shape: {anchor_arr.shape}")
    print(f"cand_arr   shape: {cand_arr.shape}")
    print(f"labels shape: {label_arr.shape}")
    print(f"parse_sims shape: {parse_sim_arr.shape}")

    # 7) save npz
    print(f"Saving to {output_npy} ...")
    np.savez(
        output_npy,
        anchor=anchor_arr,
        candidate=cand_arr,
        label=label_arr,
        parse_similarity=parse_sim_arr
    )
    print("Done.")

# -----------------------------
# main guard
if __name__ == "__main__":
    # For safety if using multiprocess,
    # import multiprocessing
    # multiprocessing.set_start_method("spawn", force=True)
    main()
