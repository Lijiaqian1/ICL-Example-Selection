# inference_with_probe.py

import argparse
import json
import torch
import torch.nn as nn
import numpy as np
import faiss
from transformers import AutoTokenizer, LlamaForCausalLM
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

##############################################################################
# 1) MLP probe model definition
#    (same architecture as in your training code)
##############################################################################

class MLPProbe(nn.Module):
    def __init__(self, hidden_dim, out_dim=1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),  
            nn.ReLU(),
            nn.Linear(256, 128),            
            nn.ReLU(),
            nn.Linear(128, 64),             
            nn.ReLU(),
            nn.Linear(64, out_dim)   
        )
    def forward(self, anchor_vec, cand_vec):
        x = torch.cat([anchor_vec, cand_vec], dim=1)  # (batch, 2*hidden_dim)
        logits = self.fc(x)  # (batch, 1)
        return logits

##############################################################################
# 2) LLaMA hidden-states extraction (batched or single)
##############################################################################

def get_llama_hidden_states(model, tokenizer, text, device, target_layer=None):
    """Single-sentence version (for small usage). 
       If you want to do big batches, adapt from earlier code."""
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    inputs = {k: v.to(device) for k,v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    all_hs = outputs.hidden_states

    if target_layer is None:
        chosen = all_hs[-1]  # final layer
    else:
        chosen = all_hs[target_layer]

    # average across seq tokens
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    sum_hs = (chosen[0] * mask[0]).sum(dim=0)
    length = mask[0].sum(dim=0)
    rep = sum_hs / length
    return rep.unsqueeze(0)  # shape => (1, hidden_dim)

##############################################################################
# 3) Main inference pipeline
##############################################################################

def main():
    # ========== A. Paths / configs ==========
    # MTop data 
    mtop_json = "../data/mtop/en/mtop_train_sampled.json"
    # MLP probe checkpoint
    mlp_checkpoint = "mlp_probe.pth"
    # LLaMA model
    llama_model_name = "meta-llama/Llama-2-7b-hf"
    access_token = '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'
    # Final output txt
    out_prompt_probe_txt = "ICL_prompt_probe.txt"
    out_prompt_embed_txt = "ICL_prompt_embedding.txt"

    parser = argparse.ArgumentParser(description="Find top-k candidates using Faiss.")
    parser.add_argument(
        "input_sentence",
        type=str,
        help="The input sentence to process.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="The number of top candidates to retrieve. Default is 5.",
    )
    args = parser.parse_args()

    new_input_sentence = args.input_sentence

    # Number of nearest neighbors
    top_k = args.top_k
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ========== B. Load MTop dataset + build Faiss index ==========

    # Load dataset
    with open(mtop_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    # dataset[i] => {"original_sentence": "...", "mtop_parsing": "..."}

    # Sentence embedding for "embedding + Faiss" approach
    st_model_name = "all-MiniLM-L6-v2"
    st_model = SentenceTransformer(st_model_name)
    print(f"Embedding entire dataset of size {len(dataset)} with {st_model_name} ...")
    all_texts = [item["original_sentence"] for item in dataset]
    embeddings = st_model.encode(all_texts, batch_size=32, show_progress_bar=True)
    embeddings = embeddings.astype(np.float32)
    d = embeddings.shape[1]

    print("Building Faiss index ...")
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    # ========== C. Load MLP probe + model + tokenizer for LLaMA ==========

    # 1) MLP probe
    # hidden_dim = 4096 if final layer of llama2-7b (or check dimension)
    # But you can do: hidden_dim = embeddings_of_llama.size(1)
    # For LLaMA-7b final hidden: around 4096
    hidden_dim = 4096  
    mlp_probe = MLPProbe(hidden_dim=hidden_dim)
    mlp_probe.load_state_dict(torch.load(mlp_checkpoint, map_location="cpu"))
    mlp_probe.to(device)
    mlp_probe.eval()

    # 2) LLaMA
    print(f"Loading LLaMA model {llama_model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(llama_model_name, token=access_token)
    model = LlamaForCausalLM.from_pretrained(
        llama_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # ========== D. Prompt building function for ICL ==========

    def build_mtop_prompt(selected_examples, new_utterance):
        """
        selected_examples => list of (sent, parse)
        new_utterance => the target user utterance
        Return a big multi-example prompt
        """
        prompt = "Below are examples of converting user utterances into Mtop semantic parses:\n"
        for i, (ex_sent, ex_parse) in enumerate(selected_examples, start=1):
            prompt += f"\nExample {i}:\nUser: {ex_sent}\nParse: {ex_parse}\n"
        prompt += "\nNow I have a new user utterance.\n"
        prompt += f"User: {new_utterance}\n"
        prompt += "LLAMA2 Parse:"
        return prompt

    # ========== E. The actual inference steps ==========

    def predict_pos_or_neg(anchor_vec, cand_vec):
        """
        Use mlp_probe to predict pos/neg. Return bool or label str
        anchor_vec, cand_vec => shape (1, hidden_dim)
        """
        mlp_probe.eval()
        with torch.no_grad():
            logits = mlp_probe(anchor_vec.to(device), cand_vec.to(device))
            # shape => (1,1)
            prob = torch.sigmoid(logits).item()  # single float
        return "pos" if prob>0.5 else "neg"

    def find_topk_candidates(user_sentence, top_k=5):
        """
        1) embed user_sentence => search top_k from Faiss
        2) returns (topk_indices, topk_scores)
        """
        user_emb = st_model.encode([user_sentence]).astype(np.float32)
        D, I = index.search(user_emb, top_k)
        # I.shape => (1, top_k)
        indices = I[0].tolist()
        # D[0]. => cos sim or IP
        return indices

    # Let's do a minimal example: user input
    #new_input_sentence = "Remind me to call my father on Sunday."

    # Step1: find top_k from embedding approach
    topk_indices = find_topk_candidates(new_input_sentence, top_k=top_k)

    # Step2: for each candidate, get hidden states of user & candidate => pass to mlp
    # => filter out pos
    print("Computing hidden states for new input sentence with LLaMA ...")
    user_rep = get_llama_hidden_states(model, tokenizer, new_input_sentence, device)
    # shape => (1, hidden_dim)

    # We'll store two sets of (sent, parse):
    # A) For "ICL w/embedding only"
    embed_exemplars = []
    # B) For "ICL w/pos predicted by MLP"
    pos_exemplars   = []

    for idx_ in topk_indices:
        cand_sent  = dataset[idx_]["original_sentence"]
        cand_parse = dataset[idx_]["mtop_parsing"]

        # get hidden for candidate
        cand_rep = get_llama_hidden_states(model, tokenizer, cand_sent, device)

        # predict with MLP
        label_ = predict_pos_or_neg(user_rep, cand_rep)
        # store example
        embed_exemplars.append( (cand_sent, cand_parse) )
        if label_ == "pos":
            pos_exemplars.append( (cand_sent, cand_parse) )

    # Step3: build two ICL prompts & save to text

    prompt_mlp  = build_mtop_prompt(pos_exemplars, new_input_sentence)
    prompt_embed= build_mtop_prompt(embed_exemplars, new_input_sentence)

    # save
    with open("ICL_prompt_probe.txt", "w", encoding="utf-8") as fw:
        fw.write(prompt_mlp)

    with open("ICL_prompt_embedding.txt", "w", encoding="utf-8") as fw:
        fw.write(prompt_embed)

    print("Done. Wrote ICL_prompt_probe.txt and ICL_prompt_embedding.txt.")


if __name__ == "__main__":
    main()
