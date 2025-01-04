import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM

def load_mtop_data(mtop_sampled_json):
    """
    Load the dataset entries => a list of dict with 'original_sentence'.
    Return them as a list of sentences for indexing.
    """
    with open(mtop_sampled_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [item["original_sentence"] for item in data]

def get_llama_hidden_states_batched(
    model,
    tokenizer,
    texts,            # list of strings
    device,
    target_layer=None
):
    """
    Batched version of hidden-state extraction for LLaMA.
    1) Tokenize the batch.
    2) Run forward with output_hidden_states=True.
    3) Extract the hidden states from the target layer or last layer.
    4) Perform average pooling across sequence dimension.
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

    mask_float = attention_mask.unsqueeze(-1).expand(chosen_hs.size()).float()
    sum_hs = (chosen_hs * mask_float).sum(dim=1)
    lengths_1d = attention_mask.sum(dim=1).unsqueeze(-1)  # shape (batch,1)
    mean_hs = sum_hs / lengths_1d  # broadcast => (batch, hidden_dim)

    return mean_hs.cpu().numpy()

def main():
    # 1) Config
    mtop_sampled_json = "../data/mtop/en/mtop_train_sampled.json"
    output_npy = "train_llama_emb.npz"
    target_layer = None  # None => last layer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8  # Adjust based on available memory

    # 2) Load data
    print("Loading dataset entries...")
    sentences = load_mtop_data(mtop_sampled_json)
    print(f"Total sentences: {len(sentences)}")

    # 3) Load LLaMA2 model
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

    # 4) Batch process sentences
    all_hidden_reps = []

    print("Extracting hidden representations...")
    for start in tqdm(range(0, len(sentences), batch_size)):
        batch_texts = sentences[start:start+batch_size]
        reps = get_llama_hidden_states_batched(
            model, tokenizer, batch_texts, device, target_layer=target_layer
        )  # shape (batch_size, hidden_dim)
        all_hidden_reps.append(reps)

    # Combine all batches into a single array
    all_hidden_reps = np.concatenate(all_hidden_reps, axis=0)  # shape (N, hidden_dim)

    print(f"Final hidden representation shape: {all_hidden_reps.shape}")

    # 5) Save to file
    print(f"Saving to {output_npy} ...")
    np.savez(output_npy, hidden=all_hidden_reps)
    print("Done.")

if __name__ == "__main__":
    main()
