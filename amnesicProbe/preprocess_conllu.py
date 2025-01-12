import json

def load_conllu(conllu_path):
    """
    Parse CoNLL-U file and extract tokens and POS.
    Save sentences as JSON for efficient reuse.
    """
    sentences = []
    current_sentence = []
    with open(conllu_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                if not line and current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue
            cols = line.split("\t")
            if len(cols) < 5:
                continue
            token_str = cols[1]
            upos = cols[3]
            current_sentence.append({"token": token_str, "pos": upos})
    if current_sentence:
        sentences.append(current_sentence)

    return sentences

def save_to_json(data, output_path):
    """
    Save preprocessed data to JSON file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def main():
    conllu_path = "../data/UD_Treebank/en_ewt-ud-train.conllu"  # Change to your CoNLL-U file path
    output_path = "preprocessed_data.json"

    print("Preprocessing CoNLL-U data...")
    sentences = load_conllu(conllu_path)
    print(f"Processed {len(sentences)} sentences.")

    print(f"Saving preprocessed data to {output_path}...")
    save_to_json(sentences, output_path)
    print("Done!")

if __name__ == "__main__":
    main()
