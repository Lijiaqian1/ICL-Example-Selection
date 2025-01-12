import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM

def load_mtop_data(mtop_sampled_json):
    with open(mtop_sampled_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data  

def load_pairs(pairs_json):
    with open(pairs_json, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    return pairs

def get_llama_hidden_states_batched(
    model,
    tokenizer,
    texts,            
    device,
    target_layer=None
):

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


    mask_float = attention_mask.unsqueeze(-1).expand(chosen_hs.size()).float()
   
    sum_hs = (chosen_hs * mask_float).sum(dim=1)
    lengths = mask_float.sum(dim=1)  
    lengths_1d = attention_mask.sum(dim=1).unsqueeze(-1)  
    mean_hs = sum_hs / lengths_1d  

    return mean_hs.cpu().numpy()

def main():
    mtop_sampled_json = "../data/mtop/en/mtop_train_sampled.json"      
    pairs_json = "../scripts/mtop_contrastive_pairs.json"                    
    output_npy = "llama2_reps.npz"
    target_layer = None 

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8  
    print("Loading dataset entries...")
    data = load_mtop_data(mtop_sampled_json)  
    print("Loading pairs (pos/neg + parse_similarity)...")
    pairs = load_pairs(pairs_json)

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

    anchor_idxs = [p["anchor_idx"] for p in pairs]
    candidate_idxs = [p["candidate_idx"] for p in pairs]
    unique_indices = set(anchor_idxs + candidate_idxs)
    unique_indices_list = sorted(list(unique_indices))

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
        )
        for i, rid in enumerate(reps):
            all_reps.append(rid)


    for i, idx_ in enumerate(idx_map):
        index_to_rep[idx_] = all_reps[i]

    
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
  
        a_vec = index_to_rep[ai]
        c_vec = index_to_rep[aj]

        anchor_reps.append(a_vec)
        cand_reps.append(c_vec)
        labels.append(lab)
     
        parse_sims.append(psim if psim is not None else np.nan)

    anchor_arr = np.stack(anchor_reps, axis=0)  
    cand_arr   = np.stack(cand_reps, axis=0)
    label_arr  = np.array(labels, dtype=object)
    parse_sim_arr = np.array(parse_sims, dtype=float)

    print(f"anchor_arr shape: {anchor_arr.shape}")
    print(f"cand_arr   shape: {cand_arr.shape}")
    print(f"labels shape: {label_arr.shape}")
    print(f"parse_sims shape: {parse_sim_arr.shape}")


    print(f"Saving to {output_npy} ...")
    np.savez(
        output_npy,
        anchor=anchor_arr,
        candidate=cand_arr,
        label=label_arr,
        parse_similarity=parse_sim_arr
    )
    print("Done.")


if __name__ == "__main__":
    main()
