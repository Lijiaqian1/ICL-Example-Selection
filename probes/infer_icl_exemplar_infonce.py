#!/usr/bin/env python3
# infer_icl_exemplars.py
"""
Usage example:
  python infer_icl_exemplars.py --top_k 5 --probe_ckpt siamese_probe.pth \
      --mtop_train ../data/mtop/en/mtop_train_sampled.json \
      --train_llama_npz train_llama_emb.npz \
      test.json

This script does:
1) Load the train dataset (`mtop_train_sampled.json`).
2) Load LLaMA hidden vectors (`train_llama_npz`) corresponding to the train dataset.
3) Load the trained Siamese probe (`siamese_probe.pth`).
4) For each test sentence from `test.json`:
   - Compute its LLaMA hidden representation.
   - Pass it through the trained Siamese network.
   - Compute distances to **all 10,000 training examples** in the aligned space.
   - Select **Top-K** closest training examples.
   - Save the results to `probe_results.json`.
"""

import sys
import json
import argparse
import numpy as np
import torch
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
        return self.net(x)  # Output shape: (batch, out_dim)

def load_probe(probe_ckpt_path, hidden_dim=4096, out_dim=256, device="cpu"):
    model = SiameseProbe(hidden_dim=hidden_dim, out_dim=out_dim)
    state = torch.load(probe_ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

###############################################################################
# 2) Load train data + LLaMA hidden vectors
###############################################################################

def load_mtop_train(mtop_train_sampled_path):
    """Load training dataset with original sentences and parsing."""
    with open(mtop_train_sampled_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data  # List of { "original_sentence":..., "mtop_parsing":... }

def get_llama_hidden_states(model, tokenizer, text, device, target_layer=None):
    """
    Extract the final hidden state of a given text using LLaMA.
    Uses mean-pooling over token embeddings.
    """
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True, padding=True, truncation=True)
    for k, v in inputs.items():
        inputs[k] = v.to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    all_hs = outputs.hidden_states

    chosen = all_hs[-1] if (target_layer is None) else all_hs[target_layer]
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    sum_hs = (chosen * mask).sum(dim=1)
    length = mask.sum(dim=1)
    rep = sum_hs / length
    return rep  # Shape: (1, hidden_dim)

###############################################################################
# 3) Inference Logic
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Infer ICL exemplars using Siamese probe in LLaMA space.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of final exemplars to pick.")
    parser.add_argument("--probe_ckpt", type=str, default="siamese_probe.pth", help="Trained siamese probe ckpt.")
    parser.add_argument("--mtop_train", type=str, default="../data/mtop/en/mtop_train_sampled.json", help="Train dataset.")
    parser.add_argument("--train_llama_npz", type=str, default="train_llama_emb.npz",
                        help="NPZ file containing LLaMA hidden vectors for each train sentence (same order).")
    parser.add_argument("input_file", type=str, nargs="?", default=None,
                        help="Test JSON file with multiple sentences.")
    args = parser.parse_args()

    top_k = args.top_k
    probe_ckpt_path = args.probe_ckpt
    train_path = args.mtop_train
    llama_train_npz = args.train_llama_npz
    input_file = args.input_file

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading train data from {train_path} ...")
    train_data = load_mtop_train(train_path)
    print(f"Loading LLaMA hidden vectors from {llama_train_npz} ...")
    loaded = np.load(llama_train_npz)
    train_llama_emb = loaded["hidden"]  # Shape: (10000, 4096)
    
    hidden_dim = train_llama_emb.shape[1]  # 4096
    out_dim = 256  

    print(f"Loading Siamese probe from {probe_ckpt_path} => hidden_dim={hidden_dim}, out_dim={out_dim}")
    probe = load_probe(probe_ckpt_path, hidden_dim=hidden_dim, out_dim=out_dim, device=device)

    llama_model_name = "meta-llama/Llama-2-7b-hf"
    print(f"Loading LLaMA model: {llama_model_name}")
    '''access_token = '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'
    tokenizer = AutoTokenizer.from_pretrained(llama_model_name, token=access_token)
    model = LlamaForCausalLM.from_pretrained(
        llama_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )'''

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16, device_map="auto")  

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Load test file
    with open(input_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    probe_out = []

    for i, item in enumerate(test_data):
        test_sentence = item["original_sentence"]

        # Compute test sentence representation
        test_h = get_llama_hidden_states(model, tokenizer, test_sentence, device)  # (1, 4096)
        test_z = probe(test_h.to(device)).detach().cpu().numpy()[0]  # (1, 256) -> (256,)

        # Compute distances to all training examples
        train_z = probe(torch.tensor(train_llama_emb, dtype=torch.float32, device=device)).detach().cpu().numpy()  # (10000, 256)
        distances = np.linalg.norm(train_z - test_z, axis=1)  # Compute L2 distance

        # Get top-K nearest indices
        top_k_indices = np.argsort(distances)[:top_k]

        probe_out.append({
            "test_idx": i,
            "probe_topk": top_k_indices.tolist()
        })

    # Save probe results
    probe_file = "probe_infonce_results.json"
    with open(probe_file, "w", encoding="utf-8") as fw:
        json.dump(probe_out, fw, ensure_ascii=False, indent=2)

    print(f"Done. Wrote {probe_file}.")


if __name__ == "__main__":
    main()
