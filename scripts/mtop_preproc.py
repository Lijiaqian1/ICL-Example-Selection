import json

def preprocess_train_file(input_file, output_file):
    # 存储处理后的数据
    result = []

    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            # 分割每一行，提取原句和 mtop parsing
            parts = line.strip().split('\t')
            if len(parts) >= 6:
                original_sentence = parts[3]
                mtop_parsing = parts[6]

                # 添加到结果列表
                result.append({
                    "original_sentence": original_sentence,
                    "mtop_parsing": mtop_parsing
                })

    # 将结果保存为 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(result, json_file, ensure_ascii=False, indent=4)

# 定义输入文件和输出文件路径
'''
input_file = "../data/mtop/en/train.txt"
output_file = "../data/mtop/en/mtop_train.json"
'''

input_file = "../data/mtop/en/test.txt"
output_file = "../data/mtop/en/mtop_test.json"

# 运行预处理函数
preprocess_train_file(input_file, output_file)

print(f"Data has been successfully saved to {output_file}")
