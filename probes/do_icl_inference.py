#!/usr/bin/env python3
# do_icl_inference.py

import argparse
import json
import os
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

###############################################################################
# 1) Prompt assembly
###############################################################################

def assemble_prompt(exemplar_list, target_sentence):
    """
    exemplar_list: list of (train_sentence, train_parse)
    target_sentence: the test query
    Return: a multi-example prompt string
    We'll add some auxiliary content + a marker for final parse.
    """
    prompt = "Below are examples of converting user utterances into MTop semantic parses:\n"
    for i, (ex_sent, ex_parse) in enumerate(exemplar_list, start=1):
        prompt += f"\nExample {i}:\nUser: {ex_sent}\nParse: {ex_parse}\n"
    prompt += "\nNow I have a new user utterance.\n"
    prompt += f"User: {target_sentence}\n"
    prompt += "LLAMA2 Parse: "  # marker
    return prompt

def extract_parse(generated_text, marker="LLAMA2 Parse:"):
    """
    Attempt to locate the parse portion in the LLaMA output.
    We'll search for the marker and extract content until the next newline.
    """
    start_idx = generated_text.find(marker)
    if start_idx == -1:
        # no marker found, fallback
        return generated_text.strip()

    parse_part = generated_text[start_idx + len(marker):]
    end_idx = parse_part.find("\n")
    if end_idx == -1:
        # no newline found, return everything after marker
        return parse_part.strip()

    return parse_part[:end_idx].strip()

###############################################################################
# 2) Main
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Run ICL with LLaMA2 for parse generation.")
    parser.add_argument("--results_file", type=str, required=True, 
                        help="Path to either embedding_results.json or probe_results.json.")
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to the test JSON (like test.json).")
    parser.add_argument("--train_file", type=str, default="../data/mtop/en/mtop_train_sampled.json",
                        help="MTop training data with original_sentence, parse.")
    parser.add_argument("--output_file", type=str, default="llama_inference_output.json",
                        help="Where to store final (idx, original_sentence, generated_parse).")
    parser.add_argument("--top_k", type=int, default=5,
                        help="We assume results_file has top_k, but let's confirm the length.")
    parser.add_argument("--llama_model", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="Hugging Face model name or path for LLaMA2.")
    parser.add_argument("--access_token", type=str, default="hf_xxx",
                        help="Your Hugging Face access token if needed.")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="Max tokens for generation.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for generation.")
    args = parser.parse_args()

    # 1) Load the results file (embedding or probe)
    with open(args.results_file, "r", encoding="utf-8") as f:
        results_data = json.load(f)

    if len(results_data) == 0:
        print("No data in results_file!")
        return

    first_keys = list(results_data[0].keys())
    candidate_key = "embedding_topk" if "embedding_topk" in first_keys else "probe_topk"

    # 2) Load test file
    with open(args.test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # 3) Load train data
    with open(args.train_file, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    # 4) Load LLaMA2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading LLaMA model {args.llama_model} ...")
    #tokenizer = AutoTokenizer.from_pretrained(args.llama_model, token=args.access_token)
    tokenizer = AutoTokenizer.from_pretrained(args.llama_model, token='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')
    model = LlamaForCausalLM.from_pretrained(
        args.llama_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # 5) For each item in results_data
    final_outputs = []

    for item in results_data:
        test_idx = item["test_idx"]
        topk_idxs = item[candidate_key]

        if test_idx < 0 or test_idx >= len(test_data):
            print(f"Warning: test_idx={test_idx} out of range. Skipping.")
            continue

        target_entry = test_data[test_idx]
        user_sentence = target_entry["original_sentence"]

        exemplars = []
        for idx_ in topk_idxs:
            if idx_ < 0 or idx_ >= len(train_data):
                continue
            train_sent = train_data[idx_]["original_sentence"]
            train_parse = train_data[idx_].get("mtop_parsing", "")
            exemplars.append((train_sent, train_parse))

        prompt_text = assemble_prompt(exemplars, user_sentence)

        inputs = tokenizer(prompt_text, return_tensors="pt")
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        with torch.no_grad():
            gen_output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=False,
                top_k=50
            )
        gen_text = tokenizer.decode(gen_output[0], skip_special_tokens=True)

        parse_result = extract_parse(gen_text, marker="LLAMA2 Parse:")

        final_outputs.append({
            "idx": test_idx,
            "original_sentence": user_sentence,
            "generated_parse": parse_result
        })

    with open(args.output_file, "w", encoding="utf-8") as fw:
        json.dump(final_outputs, fw, ensure_ascii=False, indent=2)
    print(f"Done. Wrote {args.output_file} with {len(final_outputs)} items.")

if __name__ == "__main__":
    main()
