#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

def preprocess_smcalflow(input_path, output_path, max_lines=5000):
    """
    从 smcalflow 数据集中提取用户的原始语句和对应的 lispress 解析，并保存到 JSON 文件中。
    只处理前 max_lines 行（默认 5000）。
    """
    results = []
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            #if i >= max_lines:
                #break

            try:
                dialogue_data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            turns = dialogue_data.get("turns", [])
            for turn in turns:
                user_utt = turn.get("user_utterance", {})
                original_text = user_utt.get("original_text", "").strip()
                lispress = turn.get("lispress", "").strip()
                if original_text and lispress:
                    results.append({
                        "original_sentence": original_text,
                        "parse": lispress
                    })


    with open(output_path, "w", encoding="utf-8") as fw:
        json.dump(results, fw, ensure_ascii=False, indent=2)


def main():
    #input_file = "../../data/SMCalflow/train.dataflow_dialogues_merged.jsonl"
    #output_file = "smcalflow_preprocessed.json"
    input_file = "../../data/SMCalflow/train.dataflow_dialogues_test.jsonl"
    output_file = "smcalflow_test_preprocessed.json"
    preprocess_smcalflow(input_file, output_file, max_lines=5000)
    print(f"Done! Wrote lines to {output_file} .")


if __name__ == "__main__":
    main()
