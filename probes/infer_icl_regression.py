#!/usr/bin/env python3
# infer_icl_regression.py

import argparse
import json
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, LlamaForCausalLM

###############################################################################
# 1) 回归探针模型定义
###############################################################################

class MLPRegressionProbe(torch.nn.Module):
    def __init__(self, hidden_dim, out_dim=1):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, out_dim)
        )
    def forward(self, anchor_vec, cand_vec):
        # anchor_vec shape: (batch, hidden_dim)
        # cand_vec shape:   (batch, hidden_dim)
        x = torch.cat([anchor_vec, cand_vec], dim=1)  # shape => (batch, 2*hidden_dim)
        output = self.fc(x)  # shape => (batch, 1)
        return output.squeeze(dim=1)  # shape => (batch,)

def load_regression_probe(probe_ckpt_path, hidden_dim=4096, out_dim=1, device="cpu"):
    model = MLPRegressionProbe(hidden_dim=hidden_dim, out_dim=out_dim)
    state = torch.load(probe_ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

###############################################################################
# 2) 加载训练数据并建立Faiss索引
###############################################################################

def load_mtop_train(mtop_train_sampled_path):
    with open(mtop_train_sampled_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def build_faiss_index(train_sentences):
    st_model_name = "all-MiniLM-L6-v2"
    st_model = SentenceTransformer(st_model_name)
    print(f"Embedding {len(train_sentences)} sentences with {st_model_name} ...")
    emb = st_model.encode(train_sentences, batch_size=32, show_progress_bar=True)
    emb = emb.astype(np.float32)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return st_model, index, emb

def get_topk_candidates(st_model, index, sentence, top_k):
    query = st_model.encode([sentence]).astype(np.float32)
    _, I = index.search(query, top_k)
    return I[0].tolist()

###############################################################################
# 3) LLaMA隐藏表示提取
###############################################################################

def get_llama_hidden_states(model, tokenizer, text, device, target_layer=None):
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    for k,v in inputs.items():
        inputs[k] = v.to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    all_hs = outputs.hidden_states

    chosen = all_hs[-1] if target_layer is None else all_hs[target_layer]
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    sum_hs = (chosen[0] * mask[0]).sum(dim=0)
    length = mask[0].sum(dim=0)
    rep = sum_hs / length
    return rep.unsqueeze(0)  # shape => (1, hidden_dim)

###############################################################################
# 4) 主程序
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Infer top-k exemplars using regression probe and LLaMA hidden states.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of final exemplars to select.")
    parser.add_argument("--probe_ckpt", type=str, default="mlp_regression_probe.pth", help="Path to regression probe checkpoint.")
    parser.add_argument("--mtop_train", type=str, default="../data/mtop/en/mtop_train_sampled.json", help="Path to MTop training dataset.")
    parser.add_argument("--train_llama_npz", type=str, default="train_llama_emb.npz",
                        help="NPZ file containing LLaMA hidden vectors for each train sentence.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to test JSON file.")
    args = parser.parse_args()

    top_k = args.top_k
    probe_ckpt_path = args.probe_ckpt
    train_path = args.mtop_train
    llama_train_npz = args.train_llama_npz
    test_file = args.input_file

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading train data from {train_path} ...")
    train_data = load_mtop_train(train_path)
    train_sents = [x["original_sentence"] for x in train_data]

    st_model, index, _ = build_faiss_index(train_sents)

    print(f"Loading LLaMA hidden from {llama_train_npz} ...")
    loaded = np.load(llama_train_npz)
    train_llama_emb = loaded["hidden"]
    print("Train LLaMA embedding shape:", train_llama_emb.shape)

    hidden_dim = train_llama_emb.shape[1]
    print(f"Loading regression probe from {probe_ckpt_path} with hidden_dim={hidden_dim}")
    reg_probe = load_regression_probe(probe_ckpt_path, hidden_dim=hidden_dim, out_dim=1, device=device)

    llama_model_name = "meta-llama/Llama-2-7b-hf"
    print(f"Loading LLaMA model: {llama_model_name}")
    access_token = "hf_XXXX"  # 请替换为实际token
    tokenizer = AutoTokenizer.from_pretrained(llama_model_name, token=access_token)
    model = LlamaForCausalLM.from_pretrained(
        llama_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    regression_results = []

    for i, item in enumerate(test_data):
        test_sentence = item["original_sentence"]

        # 步骤1：embedding阶段，获取top_(5*k)候选
        big_k = top_k * 5
        candidate_indices = get_topk_candidates(st_model, index, test_sentence, big_k)

        # 步骤2：使用LLaMA获取查询句子的隐藏表示
        user_h = get_llama_hidden_states(model, tokenizer, test_sentence, device)

        # 使用回归探针为每个候选计算相似度得分
        sims = []
        for idx_ in candidate_indices:
            cand_h = train_llama_emb[idx_]  # shape (hidden_dim,)
            cand_h_t = torch.tensor(cand_h, dtype=torch.float32, device=device).unsqueeze(0)
            # 计算回归探针的相似度得分
            # user_h shape => (1, hidden_dim)，cand_h_t shape => (1, hidden_dim)
            sim_score = reg_probe(user_h.to(device), cand_h_t).item()
            sims.append((idx_, sim_score))

        # 选择最高的top_k
        sims.sort(key=lambda x: x[1], reverse=True)
        topk_indices = [idx for idx, score in sims[:top_k]]

        regression_results.append({
            "test_idx": i,
            "regression_topk": topk_indices
        })

    output_file = "regression_results.json"
    with open(output_file, "w", encoding="utf-8") as fw:
        json.dump(regression_results, fw, ensure_ascii=False, indent=2)
    print(f"Finished. Results saved to {output_file}.")

if __name__ == "__main__":
    main()
