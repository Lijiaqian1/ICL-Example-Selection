import json
import argparse
import random

def main():
    parser = argparse.ArgumentParser(description="Random baseline exemplar selection.")
    parser.add_argument("--test_file", required=True, help="Path to test JSON (e.g., mtop_test_sampled_100.json).")
    parser.add_argument("--train_file", required=True, help="Path to train JSON (e.g., mtop_train_sampled.json).")
    parser.add_argument("--out_file", required=True, help="Where to save the final topK indices JSON.")
    parser.add_argument("--top_k", type=int, default=5, help="How many exemplars to select for each test sentence.")
    args = parser.parse_args()

    test_path = args.test_file
    train_path = args.train_file
    out_path = args.out_file
    top_k = args.top_k

    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    train_size = len(train_data)
    print(f"Train data size: {train_size}")

    results = []
    for i, titem in enumerate(test_data):
        random_indices = random.sample(range(train_size), min(top_k, train_size))

        results.append({
            "test_idx": i,
            "random_topk": random_indices
        })

    with open(out_path, "w", encoding="utf-8") as fw:
        json.dump(results, fw, ensure_ascii=False, indent=2)

    print(f"Finished! Wrote {len(results)} entries to {out_path}.")


if __name__ == "__main__":
    main()
