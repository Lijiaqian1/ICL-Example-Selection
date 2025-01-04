import json
import argparse
import matplotlib.pyplot as plt
from mtop_parse_sim import mtop_tree_similarity


def compute_statistics(output_data, test_data):
    similarity_scores = []
    exact_matches = 0
    total_count = len(output_data)

    for output_entry in output_data:
        idx = output_entry["idx"]
        generated_parse = output_entry["generated_parse"]

        # 找到对应的gold parse
        gold_entry = next((item for item in test_data if item["original_sentence"] == output_entry["original_sentence"]), None)
        if not gold_entry:
            print(f"Warning: No matching entry found for idx {idx} in test data.")
            continue

        gold_parse = gold_entry["mtop_parsing"]

        # 计算相似度
        similarity = mtop_tree_similarity(generated_parse, gold_parse)
        similarity_scores.append(similarity)

        # 判断是否是exact match
        if generated_parse == gold_parse:
            exact_matches += 1

    # 统计
    avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
    exact_match_rate = exact_matches / total_count if total_count > 0 else 0

    return similarity_scores, avg_similarity, exact_matches, exact_match_rate


def visualize_statistics(similarity_scores, avg_similarity, exact_match_rate):
    # 绘制相似度分数的直方图
    plt.figure(figsize=(10, 6))
    plt.hist(similarity_scores, bins=20, edgecolor='black')
    plt.title('Similarity Score Distribution')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.axvline(avg_similarity, color='red', linestyle='dashed', linewidth=1, label=f'Average: {avg_similarity:.2f}')
    plt.legend()
    plt.show()

    # 打印 exact match 统计
    print(f"\nExact Match Rate: {exact_match_rate * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Compute parse similarity and statistics.")
    parser.add_argument("--output_json", required=True, help="Path to the output JSON file.")
    parser.add_argument("--test_json", required=True, help="Path to the test JSON file.")
    args = parser.parse_args()

    # 加载 JSON 文件
    with open(args.output_json, 'r') as f:
        output_data = json.load(f)

    with open(args.test_json, 'r') as f:
        test_data = json.load(f)

    # 计算统计信息
    similarity_scores, avg_similarity, exact_matches, exact_match_rate = compute_statistics(output_data, test_data)

    # 输出结果
    print(f"Total Entries: {len(output_data)}")
    print(f"Average Similarity: {avg_similarity:.2f}")
    print(f"Exact Matches: {exact_matches}")

    # 可视化
    visualize_statistics(similarity_scores, avg_similarity, exact_match_rate)


if __name__ == "__main__":
    main()
