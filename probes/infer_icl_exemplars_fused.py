#!/usr/bin/env python3
# infer_icl_exemplars_fused.py
"""
Usage example:
  python infer_icl_exemplars_fused.py --top_k 5 --probe_ckpt siamese_probe.pth \
      --mtop_train ../data/mtop/en/mtop_train_sampled.json \
      --train_llama_npz train_llama_emb.npz \
      test.json
"""

import argparse
import json
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, LlamaForCausalLM

###############################################################################
# 1) Siamese Probe definition (保持不变)
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
        return self.net(x)

def load_probe(probe_ckpt_path, hidden_dim=4096, out_dim=256, device="cpu"):
    model = SiameseProbe(hidden_dim=hidden_dim, out_dim=out_dim)
    state = torch.load(probe_ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

###############################################################################
# 2) Load train data + build Faiss index (embedding-based)
###############################################################################

def load_mtop_train(mtop_train_sampled_path):
    with open(mtop_train_sampled_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def build_faiss_index(train_sentences):
    st_model_name = "all-MiniLM-L6-v2"
    st_model = SentenceTransformer(st_model_name)
    print(f"Embedding {len(train_sentences)} train sents with {st_model_name} ...")
    emb = st_model.encode(train_sentences, batch_size=32, show_progress_bar=True)
    emb = emb.astype(np.float32)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return st_model, index, emb

def get_topk_embedding(st_model, index, test_sentence, top_k=5):
    query = st_model.encode([test_sentence]).astype(np.float32)
    D, I = index.search(query, top_k)
    return I[0].tolist()

###############################################################################
# 3) LLaMA hidden representation with fused layers
###############################################################################

def get_llama_hidden_states_fused(model, tokenizer, text, device, layers_to_use=None):
    """
    对给定文本text，提取多个hidden层平均融合后的表示。
    - 如果layers_to_use为None，默认使用最后四层进行平均融合。
    - 返回 shape => (1, hidden_dim)
    """
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    for k, v in inputs.items():
        inputs[k] = v.to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    all_hs = outputs.hidden_states

    # 选择要融合的层
    if layers_to_use is None:
        layers = all_hs[-4:]  # 默认最后四层
    else:
        layers = [all_hs[i] for i in layers_to_use if i < len(all_hs)]

    # 平均融合多个层
    fused_hs = torch.stack(layers, dim=0).mean(dim=0)  # shape (batch, seq_len, hidden_dim)

    mask = inputs["attention_mask"].unsqueeze(-1).float()  # shape (batch, seq_len, 1)
    sum_hs = (fused_hs[0] * mask[0]).sum(dim=0)
    length = mask[0].sum(dim=0)
    rep = sum_hs / length
    return rep.unsqueeze(0)  # shape => (1, hidden_dim)

###############################################################################
# 4) The main inference pipeline
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Infer ICL exemplars using fused layer representations with Siamese probe.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of final exemplars to pick.")
    parser.add_argument("--probe_ckpt", type=str, default="siamese_probe.pth", help="Path to the trained probe checkpoint.")
    parser.add_argument("--mtop_train", type=str, default="../data/mtop/en/mtop_train_sampled.json", help="Path to the train dataset.")
    parser.add_argument("--train_llama_npz", type=str, default="train_llama_emb.npz",
                        help="NPZ with LLaMA hidden vectors for each train sentence.")
    parser.add_argument("input_file", type=str, nargs="?", default=None,
                        help="Test JSON file with multiple sentences. If not provided, run interactive mode.")
    args = parser.parse_args()

    top_k = args.top_k
    probe_ckpt_path = args.probe_ckpt
    train_path = args.mtop_train
    llama_train_npz = args.train_llama_npz
    input_file = args.input_file

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载训练数据并构建Faiss索引
    print(f"Loading train data from {train_path} ...")
    train_data = load_mtop_train(train_path)
    train_sents = [x["original_sentence"] for x in train_data]
    st_model, index, _ = build_faiss_index(train_sents)

    # 加载预计算的LLaMA隐藏向量
    print(f"Loading LLaMA hidden from {llama_train_npz} ...")
    loaded = np.load(llama_train_npz)
    train_llama_emb = loaded["hidden"]
    print("Train LLaMA embedding shape:", train_llama_emb.shape)

    # 加载Siamese探针
    hidden_dim = train_llama_emb.shape[1]
    out_dim = 256
    print(f"Loading probe from {probe_ckpt_path} with hidden_dim={hidden_dim}, out_dim={out_dim}")
    probe = load_probe(probe_ckpt_path, hidden_dim=hidden_dim, out_dim=out_dim, device=device)

    # 加载LLaMA模型和tokenizer用于新输入
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

    if input_file is None:
        # Interactive模式
        while True:
            user_in = input("Enter a sentence (or 'quit'): ")
            if user_in.strip().lower() == "quit":
                break
            handle_single(user_in, top_k, st_model, index, train_data, train_llama_emb, probe, 
                          model, tokenizer, device)
    else:
        # 批量处理模式
        with open(input_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)
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

            # 使用融合后的隐藏向量
            user_h = get_llama_hidden_states_fused(model, tokenizer, s, device)
            user_z = probe(user_h.to(device)).detach().cpu().numpy()[0]

            results_dist = []
            for idx_ in bigger_idxs:
                cand_h = train_llama_emb[idx_]  # LLaMA hidden from npz
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

        with open("embedding_results.json", "w", encoding="utf-8") as fw:
            json.dump(embed_out, fw, ensure_ascii=False, indent=2)
        with open("probe_results.json", "w", encoding="utf-8") as fw:
            json.dump(probe_out, fw, ensure_ascii=False, indent=2)
        print("Done. Wrote embedding_results.json and probe_results.json.")


def handle_single(sentence, top_k, st_model, index, train_data, train_llama_emb, probe, model, tokenizer, device):
    embed_idxs = get_topk_embedding(st_model, index, sentence, top_k=top_k)
    print(f"\n=== Embedding top-{top_k} for: {sentence} ===")
    for idx_ in embed_idxs:
        print(f"{idx_} => {train_data[idx_]['original_sentence']}")

    big_k = top_k * 5
    bigger_idxs = get_topk_embedding(st_model, index, sentence, top_k=big_k)

    user_h = get_llama_hidden_states_fused(model, tokenizer, sentence, device)
    user_z = probe(user_h.to(device)).detach().cpu().numpy()[0]

    results_dist = []
    for idx_ in bigger_idxs:
        cand_h = train_llama_emb[idx_]
        cand_h_t = torch.tensor(cand_h, dtype=torch.float32, device=device).unsqueeze(0)
        cand_z_t = probe(cand_h_t)
        cand_z = cand_z_t.detach().cpu().numpy()[0]
        dist = np.linalg.norm(user_z - cand_z)
        results_dist.append((idx_, dist))

    results_dist.sort(key=lambda x: x[1])
    final_idxs = [x[0] for x in results_dist[:top_k]]

    print(f"\n=== Probe top-{top_k} (from 5x embedding topK) ===")
    for idx_ in final_idxs:
        print(f"{idx_} => {train_data[idx_]['original_sentence']}")
    print("")

if __name__ == "__main__":
    main()
