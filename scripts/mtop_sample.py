import json
import random
import argparse
import os

def sample_and_save(input_json_path, output_json_path, sample_size):
    """
    从 input_json_path 中随机取样 sample_size 个样本，保存到 output_json_path。
    """
    # 读取原始 JSON 数据
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_samples = len(data)
    print(f"Total samples available: {total_samples}")

    # 确保 sample_size 不超过总数
    if sample_size > total_samples:
        raise ValueError(f"Sample size {sample_size} exceeds total available samples {total_samples}.")

    # 随机采样
    sampled_data = random.sample(data, sample_size)
    print(f"Randomly sampled {sample_size} samples.")

    # 保存到新的 JSON 文件
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=2)
    print(f"Sampled data saved to {output_json_path}.")

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Randomly sample a subset of JSON data.")
    parser.add_argument("input_json", type=str, help="Path to the input JSON file.")
    parser.add_argument("sample_size", type=int, help="Number of samples to randomly select.")
    parser.add_argument("output_dir", type=str, help="Directory to save the output JSON file.")

    # 解析参数
    args = parser.parse_args()
    input_json = args.input_json
    sample_size = args.sample_size
    output_dir = args.output_dir

    # 自动生成输出文件名
    input_basename = os.path.basename(input_json)
    input_name, input_ext = os.path.splitext(input_basename)
    output_json = os.path.join(output_dir, f"{input_name}_sampled_{sample_size}{input_ext}")

    # 调用采样函数
    sample_and_save(input_json, output_json, sample_size)
