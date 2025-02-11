#!/usr/bin/env python3
import sys
import json
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

###############################################################################
# 1) Dual Encoder Probe with Interaction (Concat Mode)
###############################################################################

class DualEncoderProbe(torch.nn.Module):
    def __init__(self, hidden_dim=4096, out_dim=256, interaction_mode="concat"):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, out_dim)
        )
        self.interaction_mode = interaction_mode

        if interaction_mode == "concat":
            self.interaction_layer = torch.nn.Sequential(
                torch.nn.Linear(2 * out_dim, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 1)  # Scalar score
            )

    def forward(self, x1, x2=None):
        z1 = self.encoder(x1)
        if x2 is None:
            return z1  # 仅计算投影向量
        
        if self.interaction_mode == "concat":
            combined = torch.cat([z1, x2], dim=-1)  # Concatenate vectors
            score = self.interaction_layer(combined)  # MLP for final score
            return score.squeeze(-1)  # (batch, 1) -> (batch,)
        return None  # Invalid interaction mode

###############################################################################
# 2) Load Train Data + LLaMA Hidden Vectors
###############################################################################

def load_mtop_train(mtop_train_sampled_path):
    """Load training dataset with original sentences and parsing."""
    with open(mtop_train_sampled_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data  # List of { "original_sentence":..., "mtop_parsing":... }

def get_llama_hidden_states(model, tokenizer, text, device, target_layer=None):
    """Extract the final hidden state of a given text using LLaMA."""
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
# 3) Inference Logic (Optimized Concat)
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Infer ICL exemplars using Dual Encoder (Concat).")
    parser.add_argument("--top_k", type=int, default=5, help="Number of final exemplars to pick.")
    parser.add_argument("--probe_ckpt", type=str, default="dual_encoder.pth", help="Trained dual encoder ckpt.")
    parser.add_argument("--mtop_train", type=str, default="../data/mtop/en/mtop_train_sampled.json", help="Train dataset.")
    parser.add_argument("--train_llama_npz", type=str, default="train_llama_emb.npz",
                        help="NPZ file containing LLaMA hidden vectors for each train sentence (same order).")
    parser.add_argument("input_file", type=str, nargs="?", default=None, help="Test JSON file.")
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
    train_llama_emb = torch.tensor(loaded["hidden"], dtype=torch.float32, device=device)  # Shape: (10000, 4096)

    hidden_dim = train_llama_emb.shape[1]  # 4096
    out_dim = 256  

    print(f"Loading Dual Encoder from {probe_ckpt_path} => hidden_dim={hidden_dim}, out_dim={out_dim}")
    probe = DualEncoderProbe(hidden_dim=hidden_dim, out_dim=out_dim, interaction_mode="concat").to(device)
    probe.load_state_dict(torch.load(probe_ckpt_path, map_location=device))
    probe.eval()

    print(f"Projecting all train sentences into aligned space...")
    with torch.no_grad():
        train_z = probe(train_llama_emb)  # Shape: (10000, 256)

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
        test_z = probe(test_h.to(device))  # (1, 256)

        # Batch Concatenation: Expand test_z to match train_z shape
        B = train_z.shape[0]  # 10000
        test_expanded = test_z.expand(B, -1)  # (10000, 256)

        # Batch Compute MLP Score
        with torch.no_grad():
            scores = probe.interaction_layer(torch.cat([test_expanded, train_z], dim=-1))  # (10000, 1)
            scores = scores.squeeze(-1).cpu().numpy()  # Convert to (10000,)

        # Get top-K indices based on highest scores
        top_k_indices = np.argsort(-scores)[:top_k]

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
