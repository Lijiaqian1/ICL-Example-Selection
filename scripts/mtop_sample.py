import json
import random
import argparse
import os

def sample_and_save(input_json_path, output_json_path, sample_size):

    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_samples = len(data)
    print(f"Total samples available: {total_samples}")

    if sample_size > total_samples:
        raise ValueError(f"Sample size {sample_size} exceeds total available samples {total_samples}.")

    sampled_data = random.sample(data, sample_size)
    print(f"Randomly sampled {sample_size} samples.")

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=2)
    print(f"Sampled data saved to {output_json_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Randomly sample a subset of JSON data.")
    parser.add_argument("input_json", type=str, help="Path to the input JSON file.")
    parser.add_argument("sample_size", type=int, help="Number of samples to randomly select.")
    parser.add_argument("output_dir", type=str, help="Directory to save the output JSON file.")

    args = parser.parse_args()
    input_json = args.input_json
    sample_size = args.sample_size
    output_dir = args.output_dir

    input_basename = os.path.basename(input_json)
    input_name, input_ext = os.path.splitext(input_basename)
    output_json = os.path.join(output_dir, f"{input_name}_sampled_{sample_size}{input_ext}")

    sample_and_save(input_json, output_json, sample_size)
