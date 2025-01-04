#!/usr/bin/env python3
# infer_icl_exemplars.py
"""
Usage example:
  python infer_icl_exemplars.py --top_k 5 --probe_ckpt siamese_probe.pth \
      --mtop_train ../data/mtop/en/mtop_train_sampled.json \
      --train_llama_npz train_llama_emb.npz \
      test.json

This script does:
1) Load the train dataset (`mtop_train_sampled.json`) + a FAISS index built on a *sentence-transformers* embedding.
2) Load a separate `.npz` (`train_llama_npz`) containing final-layer LLaMA hidden vectors for each train item.
3) Load a Siamese probe (`siamese_probe.pth`) that maps LLaMA hidden vectors -> new space.
4) For each test sentence (from `test.json` if provided), do:
   - a) Use sentence-transformer + Faiss => top_K (embedding_results)
   - b) Actually retrieve top_(5*K) from embedding, then re-rank them using the Siamese probe in LLaMA hidden space => final top_K => (probe_results)
   - c) Save them in two JSON files for all test items.
If no test file is given, script runs an interactive single-sentence loop.
"""

import sys
import os
import json
import argparse
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, LlamaForCausalLM

###############################################################################
# 1) Siamese Probe definition
###############################################################################

class SiameseProbe(torch.nn.Module):
    def __init__(self, hidden_dim=4096, out_dim=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, out_dim)
        )
    def forward(self, x):
        # x => (batch, hidden_dim)
        return self.net(x)  # => (batch, out_dim)

def load_probe(probe_ckpt_path, hidden_dim=4096, out_dim=256, device="cpu"):
    model = SiameseProbe(hidden_dim=hidden_dim, out_dim=out_dim)
    state = torch.load(probe_ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

###############################################################################
# 2) Load train data + build Faiss index over sentence embeddings
###############################################################################

def load_mtop_train(mtop_train_sampled_path):
    with open(mtop_train_sampled_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data  # list of { "original_sentence":..., "mtop_parsing":... }

def build_faiss_index(train_sentences):
    """
    Build a Faiss index using a SentenceTransformer embedding 
    to do fast top-K retrieval at inference time.
    """
    st_model_name = "all-MiniLM-L6-v2"
    st_model = SentenceTransformer(st_model_name)
    print(f"Embedding {len(train_sentences)} train sents with {st_model_name} ...")
    emb = st_model.encode(train_sentences, batch_size=32, show_progress_bar=True)
    emb = emb.astype(np.float32)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return st_model, index, emb

def get_topk_embedding(st_model, index, test_sentence, top_k=5):
    """Return top_k indices from the train set by sentence-embedding approach."""
    query = st_model.encode([test_sentence]).astype(np.float32)
    D, I = index.search(query, top_k)
    return I[0].tolist()

###############################################################################
# 3) LLaMA hidden representation for "new input"
#    We'll do real LLaMA forward pass => final hidden
###############################################################################

def get_llama_hidden_states(model, tokenizer, text, device, target_layer=None):
    """
    Single-sentence version for LLaMA hidden state extraction.
    - Run forward with output_hidden_states=True
    - If target_layer is None => use last layer
    - Average across tokens.
    returns shape => (1, hidden_dim)
    """
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    for k,v in inputs.items():
        inputs[k] = v.to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    all_hs = outputs.hidden_states

    chosen = all_hs[-1] if (target_layer is None) else all_hs[target_layer]
    mask = inputs["attention_mask"].unsqueeze(-1).float()  # (1, seq_len, 1)
    sum_hs = (chosen[0] * mask[0]).sum(dim=0)
    length = mask[0].sum(dim=0)
    rep = sum_hs / length
    return rep.unsqueeze(0)  # shape => (1, hidden_dim)

###############################################################################
# The main inference pipeline
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Infer ICL exemplars using Faiss + Siamese probe in LLaMA space.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of final exemplars to pick.")
    parser.add_argument("--probe_ckpt", type=str, default="siamese_probe.pth", help="Trained siamese probe ckpt.")
    parser.add_argument("--mtop_train", type=str, default="../data/mtop/en/mtop_train_sampled.json", help="Train dataset.")
    parser.add_argument("--train_llama_npz", type=str, default="train_llama_emb.npz",
                        help="NPZ file containing LLaMA hidden vectors for each train sentence (same order).")
    parser.add_argument("input_file", type=str, nargs="?", default=None,
                        help="Test JSON file with multiple sentences. If not provided, script runs single interactive mode.")
    args = parser.parse_args()

    top_k = args.top_k
    probe_ckpt_path = args.probe_ckpt
    train_path = args.mtop_train
    llama_train_npz = args.train_llama_npz
    input_file = args.input_file

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load train data + build Faiss index (embedding-based)
    print(f"Loading train data from {train_path} ...")
    train_data = load_mtop_train(train_path)  # list of dict
    train_sents = [x["original_sentence"] for x in train_data]

    # build faiss index
    st_model, index, _ = build_faiss_index(train_sents)

    # 2) Load pre-computed LLaMA final hidden for each train sentence (train_llama_emb.npz)
    # shape => (N, hidden_dim)
    print(f"Loading LLaMA hidden from {llama_train_npz} ...")
    loaded = np.load(llama_train_npz)
    train_llama_emb = loaded["hidden"]  # e.g. shape (N, 4096)
    print("Train LLaMA embedding shape:", train_llama_emb.shape)

    # 3) Load the siamese probe
    hidden_dim = train_llama_emb.shape[1]  # e.g. 4096
    out_dim = 256
    print(f"Loading siamese probe from {probe_ckpt_path} => hidden_dim={hidden_dim}, out_dim={out_dim}")
    probe = load_probe(probe_ckpt_path, hidden_dim=hidden_dim, out_dim=out_dim, device=device)

    # 4) Load LLaMA model for test input
    # if we want real LLaMA hidden for new queries
    llama_model_name = "meta-llama/Llama-2-7b-hf"
    print(f"Loading LLaMA model: {llama_model_name}")
    access_token = "hf_XXXX"
    tokenizer = AutoTokenizer.from_pretrained(llama_model_name, token=access_token)
    model = LlamaForCausalLM.from_pretrained(
        llama_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Interactive or batch
    if input_file is None:
        while True:
            user_in = input("Enter a sentence (or 'quit'): ")
            if user_in.strip().lower() == "quit":
                break
            handle_single(user_in, top_k, st_model, index, train_data, train_llama_emb, probe, 
                          model, tokenizer, device)
    else:
        with open(input_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        # We'll produce two JSON:
        embed_out = []
        probe_out = []

        for i, item in enumerate(test_data):
            s = item["original_sentence"]
            # Step A: top_k from embedding
            embed_idxs = get_topk_embedding(st_model, index, s, top_k=top_k)
            embed_out.append({
                "test_idx": i,
                "embedding_topk": embed_idxs
            })

            # Step B: top_(5*k)
            big_k = top_k*5
            bigger_idxs = get_topk_embedding(st_model, index, s, top_k=big_k)

            # user hidden from LLaMA
            user_h = get_llama_hidden_states(model, tokenizer, s, device)
            # map user => new space
            user_z = probe(user_h.to(device)).detach().cpu().numpy()[0]

            # re-rank bigger_idxs in probe space
            results_dist = []
            for idx_ in bigger_idxs:
                # candidate hidden from LLaMA npz
                cand_h = train_llama_emb[idx_]  # shape (4096,)
                cand_h_t = torch.tensor(cand_h, dtype=torch.float32, device=device).unsqueeze(0)
                cand_z_t = probe(cand_h_t)
                cand_z = cand_z_t.detach().cpu().numpy()[0]
                dist = np.linalg.norm(user_z - cand_z)
                results_dist.append((idx_, dist))
            results_dist.sort(key=lambda x: x[1])
            final_idxs = [x[0] for x in results_dist[:top_k]]

            probe_out.append({
                "test_idx": i,
                "probe_topk": final_idxs
            })

        # Save
        embed_file = "embedding_results.json"
        probe_file = "probe_results.json"
        with open(embed_file, "w", encoding="utf-8") as fw:
            json.dump(embed_out, fw, ensure_ascii=False, indent=2)
        with open(probe_file, "w", encoding="utf-8") as fw:
            json.dump(probe_out, fw, ensure_ascii=False, indent=2)
        print(f"Done. Wrote {embed_file} and {probe_file}.")


def handle_single(sentence, top_k, st_model, index, train_data, train_llama_emb, probe, model, tokenizer, device):
    """
    Interactive single-sentence usage.
    1) get topK from embedding => print
    2) get top(5*K) => re-rank with siamese probe in LLaMA space => print
    """
    embed_idxs = get_topk_embedding(st_model, index, sentence, top_k=top_k)
    print(f"\n=== Embedding top-{top_k} for: {sentence} ===")
    for idx_ in embed_idxs:
        print(f"{idx_} => {train_data[idx_]['original_sentence']}")

    big_k = top_k * 5
    bigger_idxs = get_topk_embedding(st_model, index, sentence, top_k=big_k)

    # get user hidden from llama
    user_h = get_llama_hidden_states(model, tokenizer, sentence, device)
    user_z = probe(user_h.to(device)).detach().cpu().numpy()[0]

    results_dist = []
    for idx_ in bigger_idxs:
        cand_h = train_llama_emb[idx_]  # shape (4096,)
        cand_h_t = torch.tensor(cand_h, dtype=torch.float32, device=device).unsqueeze(0)
        cand_z_t = probe(cand_h_t)
        cand_z = cand_z_t.detach().cpu().numpy()[0]
        dist = np.linalg.norm(user_z - cand_z)
        results_dist.append((idx_, dist))

    results_dist.sort(key=lambda x: x[1])
    final_idxs = [x[0] for x in results_dist[:top_k]]

    print(f"\n=== Probe top-{top_k} (re-ranked from 5x embedding topK) ===")
    for idx_ in final_idxs:
        print(f"{idx_} => {train_data[idx_]['original_sentence']}")
    print("")


if __name__ == "__main__":
    main()
