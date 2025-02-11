import argparse
import json
import os
import openai

def assemble_prompt(exemplar_list, target_sentence):
    """构造 ICL Prompt"""
    prompt = "Below are examples of converting user utterances into MTop semantic parses:\n"
    for i, (ex_sent, ex_parse) in enumerate(exemplar_list, start=1):
        prompt += f"\nExample {i}:\nUser: {ex_sent}\nParse: {ex_parse}\n"
    prompt += "\nNow I have a new user utterance.\n"
    prompt += f"User: {target_sentence}\n"
    prompt += "GPT-4o Parse: "
    return prompt

def extract_parse_by_bracket_balance(generated_text, marker="GPT-4o Parse:"):
    """从 GPT-4o 生成的文本中提取 Semantic Parse"""
    start_idx = generated_text.find(marker)
    if start_idx == -1:
        return generated_text.strip()

    parse_part = generated_text[start_idx + len(marker):].strip()
    
    stack = []
    extracted = ""
    for char in parse_part:
        extracted += char
        if char in ["(", "[", "{"]:
            stack.append(char)
        elif char in [")", "]", "}"]:
            if stack:
                stack.pop()

        if not stack and any(bracket in extracted for bracket in [")", "]", "}"]):
            break
    return extracted.strip()

def main():
    parser = argparse.ArgumentParser(description="Run ICL with OpenAI GPT-4o for parse generation.")
    parser.add_argument("--results_file", type=str, required=True, 
                        help="Path to probe_results.json (containing exemplar indices).")
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to the test JSON (like test.json).")
    parser.add_argument("--train_file", type=str, default="../data/mtop/en/mtop_train_sampled.json",
                        help="MTop training data with original_sentence, parse.")
    parser.add_argument("--output_file", type=str, default="openai_inference_output.json",
                        help="Where to store final (idx, original_sentence, generated_parse).")
    parser.add_argument("--top_k", type=int, default=5,
                        help="We assume results_file has top_k, but let's confirm the length.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for GPT-4o generation.")
    parser.add_argument("--max_tokens", type=int, default=100,
                        help="Max tokens for generation.")
    args = parser.parse_args()

    # 读取环境变量中的 API Key
    API_KEY = os.getenv("OPENAI_API_KEY")
    if API_KEY is None:
        print("ERROR: OpenAI API Key is missing! Set OPENAI_API_KEY as an environment variable.")
        return

    client = openai.OpenAI(api_key=API_KEY)

    # 加载数据
    with open(args.results_file, "r", encoding="utf-8") as f:
        results_data = json.load(f)

    if len(results_data) == 0:
        print("No data in results_file!")
        return

    filename = os.path.basename(args.results_file)  
    prefix = filename.split("_")[0]                
    candidate_key = prefix + "_topk"            

    with open(args.test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    with open(args.train_file, "r", encoding="utf-8") as f:
        train_data = json.load(f)

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

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that converts user utterances into structured semantic parses."},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )

            gen_text = response.choices[0].message.content
            parse_result = extract_parse_by_bracket_balance(gen_text, marker="GPT-4o Parse:")

            final_outputs.append({
                "idx": test_idx,
                "original_sentence": user_sentence,
                "generated_parse": parse_result
            })

        except Exception as e:
            print(f"API request failed for test_idx={test_idx}: {str(e)}")
            continue

    # 存储结果
    with open(args.output_file, "w", encoding="utf-8") as fw:
        json.dump(final_outputs, fw, ensure_ascii=False, indent=2)
    print(f"Done. Wrote {args.output_file} with {len(final_outputs)} items.")

if __name__ == "__main__":
    main()
