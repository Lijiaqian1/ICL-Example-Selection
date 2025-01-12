import torch
import json
import numpy as np
from transformers import AutoTokenizer, LlamaForCausalLM
from tqdm import tqdm

def extract_llama_token_reps(llama_model_name, sentences, device="cuda"):
    """
    Extract LLaMA hidden representations for tokens.
    """
    tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
    model = LlamaForCausalLM.from_pretrained(
        llama_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_reps = []
    for sent in tqdm(sentences, desc="Extracting LLaMA hidden states"):
        text_str = " ".join([t["token"] for t in sent])
        inputs = tokenizer(text_str, return_tensors="pt", add_special_tokens=True)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        final_hs = outputs.hidden_states[-1][0]  # (seq_len, hidden_dim)

        sent_reps = []
        for i, token in enumerate(sent):
            if i + 1 < final_hs.size(0):  # Skip <s>
                sent_reps.append({
                    "token": token["token"],
                    "pos": token["pos"],
                    "vector": final_hs[i + 1].cpu().numpy()
                })
        all_reps.append(sent_reps)
    return all_reps

def save_to_npz(sent_reps, output_path):
    """
    Save hidden representations and POS to .npz file.
    """
    X, Y = [], []
    for sent in sent_reps:
        for tok in sent:
            X.append(tok["vector"])
            Y.append(tok["pos"])
    np.savez(output_path, hidden=np.array(X), pos=Y)
    print(f"Saved hidden representations to {output_path}")



def main():
    input_json = "preprocessed_data.json"
    output_npz = "hidden_representations.npz"
    llama_model_name = "meta-llama/Llama-2-7b-hf"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading preprocessed data from {input_json}...")
    with open(input_json, "r", encoding="utf-8") as f:
        sentences = json.load(f)

    print("Extracting hidden representations...")
    sent_reps = extract_llama_token_reps(llama_model_name, sentences, device=device)

    print(f"Saving hidden representations to {output_npz}...")
    save_to_npz(sent_reps, output_npz)
    print("Done!")

if __name__ == "__main__":
    main()
