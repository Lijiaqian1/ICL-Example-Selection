import os


input_files = [
    "train.dataflow_dialogues_sampled1.jsonl",
    "train.dataflow_dialogues_sampled2.jsonl",
    "train.dataflow_dialogues_sampled3.jsonl",
    "train.dataflow_dialogues_sampled4.jsonl",
    "train.dataflow_dialogues_sampled5.jsonl"
]
output_file = "train.dataflow_dialogues_merged.jsonl"


with open(output_file, "w") as outfile:
    for input_file in input_files:
        if os.path.exists(input_file):
            with open(input_file, "r") as infile:
                for line in infile:
                    outfile.write(line)
        else:
            print(f"Warning: {input_file} does not exist.")

print(f"Files merged into {output_file}")
