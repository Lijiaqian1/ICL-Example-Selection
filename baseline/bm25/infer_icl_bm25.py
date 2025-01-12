import json
import argparse
from rank_bm25 import BM25Okapi
import string

def tokenize(text):

    text = text.lower()
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    tokens = text.split()
    return tokens

def main():
    parser = argparse.ArgumentParser(description="BM25 baseline exemplar selection.")
    parser.add_argument("--test_file", required=True, help="Path to test JSON (e.g., mtop_test_sampled_100.json).")
    parser.add_argument("--train_file", required=True, help="Path to train JSON (e.g., mtop_train_sampled.json).")
    parser.add_argument("--out_file", required=True, help="Where to save the final topK indices JSON.")
    parser.add_argument("--top_k", type=int, default=5, help="How many exemplars to select for each test sentence.")
    args = parser.parse_args()

    test_path = args.test_file
    train_path= args.train_file
    out_path  = args.out_file
    top_k     = args.top_k

    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    train_sentences = [item["original_sentence"] for item in train_data]
    tokenized_corpus = [tokenize(s) for s in train_sentences]

    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"Done. Corpus size: {len(tokenized_corpus)}")


    results = []
    for i, titem in enumerate(test_data):
        test_sentence = titem["original_sentence"]
        query_tokens = tokenize(test_sentence)
        scores = bm25.get_scores(query_tokens)  
        top_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]

        results.append({
            "test_idx": i,
            "bm25_topk": top_indices
        })

    with open(out_path, "w", encoding="utf-8") as fw:
        json.dump(results, fw, ensure_ascii=False, indent=2)

    print(f"Finished! Wrote {len(results)} entries to {out_path}.")


if __name__ == "__main__":
    main()
