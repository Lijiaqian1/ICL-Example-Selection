import json

def preprocess_train_file(input_file, output_file):
    result = []

    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) >= 6:
                original_sentence = parts[3]
                mtop_parsing = parts[6]

                result.append({
                    "original_sentence": original_sentence,
                    "mtop_parsing": mtop_parsing
                })


    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(result, json_file, ensure_ascii=False, indent=4)


'''
input_file = "../data/mtop/en/train.txt"
output_file = "../data/mtop/en/mtop_train.json"
'''

input_file = "../data/mtop/en/test.txt"
output_file = "../data/mtop/en/mtop_test.json"


preprocess_train_file(input_file, output_file)

print(f"Data has been successfully saved to {output_file}")
