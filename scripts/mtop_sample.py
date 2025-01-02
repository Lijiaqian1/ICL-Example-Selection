import json
import random

def sample_and_save(input_json_path, output_json_path, sample_size=10000):
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

# 示例用法
if __name__ == "__main__":
    input_json = "../data/mtop/en/mtop_train.json"  # 原始 JSON 文件路径
    output_json = "mtop_train_sampled.json"  # 输出采样后的 JSON 文件路径
    SAMPLE_SIZE = 10000  # 采样数

    sample_and_save(input_json, output_json, SAMPLE_SIZE)

    # 之后可以在 `output_json` 文件中进行后续操作
