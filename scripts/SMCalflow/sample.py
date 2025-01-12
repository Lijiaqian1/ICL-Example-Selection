import json
import random

def sample_json(input_file, output_file, sample_size=10000):
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError("The input JSON file must contain a list of objects.")

        sample_size = min(sample_size, len(data))
        sampled_data = random.sample(data, sample_size)

        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(sampled_data, file, ensure_ascii=False, indent=4)

        print(f"Successfully sampled {sample_size} items and saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

input_file = './smcalflow_test_preprocessed.json'  
output_file = './smcalflow_test_sampled_100.json' 
sample_json(input_file, output_file,100)