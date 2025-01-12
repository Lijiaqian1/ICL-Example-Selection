import json

def fix_bracket_imbalance(parse_str):
    """修复括号不对称问题"""
    if not parse_str:
        return parse_str

    open_count = parse_str.count('[')
    close_count = parse_str.count(']')

    if open_count > close_count:
        parse_str += ']' * (open_count - close_count)
    elif close_count > open_count:
        parse_str = '[' * (close_count - open_count) + parse_str

    return parse_str

def process_json(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            data = json.load(file)

        for entry in data:
            if "generated_parse" in entry:
                entry["generated_parse"] = fix_bracket_imbalance(entry["generated_parse"])

        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

        print(f"Processed file saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

# 使用示例
input_file = './llama_inference_random_output_1000.json'  # 替换为你的输入文件路径
output_file = 'processed_llama_inference_random_output_1000.json'  # 替换为你的输出文件路径
process_json(input_file, output_file)
