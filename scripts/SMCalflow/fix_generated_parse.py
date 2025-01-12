import json
import argparse

def balance_parentheses(lispress_str):
    """Fix unbalanced parentheses in a Lispress string."""
    open_count = lispress_str.count("(")
    close_count = lispress_str.count(")")

    if open_count > close_count:
        lispress_str += ")" * (open_count - close_count)

    elif close_count > open_count:
        lispress_str = "(" * (close_count - open_count) + lispress_str

    return lispress_str

def fix_unbalanced_parses(input_json_path, output_json_path):
    """Fix unbalanced parentheses in generated parses from the JSON file."""
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    fixed_entries = []
    fixed_count = 0

    for entry in data:
        generated_parse = entry.get("generated_parse", "")
        open_count = generated_parse.count("(")
        close_count = generated_parse.count(")")

        if open_count != close_count:
            fixed_count += 1
            entry["generated_parse"] = balance_parentheses(generated_parse)

        fixed_entries.append(entry)

    with open(output_json_path, 'w') as f:
        json.dump(fixed_entries, f, indent=4)

    print(f"Total entries processed: {len(data)}")
    print(f"Entries with fixed parentheses: {fixed_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix unbalanced parentheses in generated parses from JSON.")
    parser.add_argument("--input_json", required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_json", required=True, help="Path to the output JSON file.")
    args = parser.parse_args()

    fix_unbalanced_parses(args.input_json, args.output_json)
