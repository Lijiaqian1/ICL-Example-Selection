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


        gold_entry = next((item for item in test_data if item["original_sentence"] == output_entry["original_sentence"]), None)
        if not gold_entry:
            print(f"Warning: No matching entry found for idx {idx} in test data.")
            continue

        gold_parse = gold_entry["mtop_parsing"]


        similarity = mtop_tree_similarity(generated_parse, gold_parse)
        similarity_scores.append(similarity)

        if generated_parse == gold_parse:
            exact_matches += 1

    avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
    exact_match_rate = exact_matches / total_count if total_count > 0 else 0

    return similarity_scores, avg_similarity, exact_matches, exact_match_rate





def main():
    parser = argparse.ArgumentParser(description="Compute parse similarity and statistics.")
    parser.add_argument("--output_json", required=True, help="Path to the output JSON file.")
    parser.add_argument("--test_json", required=True, help="Path to the test JSON file.")
    args = parser.parse_args()


    with open(args.output_json, 'r') as f:
        output_data = json.load(f)

    with open(args.test_json, 'r') as f:
        test_data = json.load(f)

    similarity_scores, avg_similarity, exact_matches, exact_match_rate = compute_statistics(output_data, test_data)


    print(f"Total Entries: {len(output_data)}")
    print(f"Average Similarity: {avg_similarity:.2f}")
    print(f"Exact Matches: {exact_matches}")



if __name__ == "__main__":
    main()
