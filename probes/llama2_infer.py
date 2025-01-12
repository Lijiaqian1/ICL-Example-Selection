import torch
import argparse
from transformers import LlamaForCausalLM, AutoTokenizer

def main():
    model_name = "meta-llama/Llama-2-7b-hf"

    print(f"Loading LLaMA2-7B from {model_name} ...")
    access_token = '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'
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",

    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    '''
    test_sentence = [

        "I'm looking for chicken recipes"

    ]

    examples = [
    {
        "input": "Has Angelika Kratzer video messaged me?",
        "parse": "[IN:GET_MESSAGE [SL:CONTACT Angelika Kratzer ] [SL:TYPE_CONTENT video ] [SL:RECIPIENT me ] ]"
    },
    {
        "input": "Maricopa County weather forecast for this week",
        "parse": "[IN:GET_WEATHER [SL:LOCATION Maricopa County ] [SL:DATE_TIME for this week ] ]"
    },
    {
        "input": "When will my next alarm start",
        "parse": "[IN:GET_ALARM [SL:ORDINAL next ] ]"
    },
    {
        "input": "text Matthew and Helen that are you free",
        "parse": "[IN:SEND_MESSAGE [SL:RECIPIENT Matthew ] [SL:RECIPIENT Helen ] [SL:CONTENT_EXACT are you free ] ]"
    },
    {
        "input": "what ingredients do is have left",
        "parse": "[IN:GET_INFO_RECIPES [SL:RECIPES_ATTRIBUTE ingredients ] ]"
    },
    {
        "input": "I am no longer available",
        "parse": "[IN:SET_UNAVAILABLE ]"
    },
    {
        "input": "Cancel my reminder about my dentist appointment",
        "parse": "[IN:DELETE_REMINDER [SL:PERSON_REMINDED my ] [SL:TODO my dentist appointment ] ]"
    },
    {
        "input": "I would like a news update.",
        "parse": "[IN:GET_STORIES_NEWS [SL:NEWS_TYPE news ] ]"
    }
]

    

    def build_prompt(examples, new_utterance):
        prompt = "Below are examples of converting user utterances into Mtop semantic parses:\n"
        for i, ex in enumerate(examples, start=1):
            prompt += f"\nExample {i}:\nUser: {ex['input']}\nParse: {ex['parse']}\n"
        prompt += "\nNow I have a new user utterance.\n"
        prompt += f"User: {new_utterance}\n"
        prompt += "LLAMA2 Parse:"
        return prompt


    prompt_text = build_prompt(examples, test_sentence)
    '''
    parser = argparse.ArgumentParser(description="Read a text file and assign its content to a variable.")
    parser.add_argument(
        "file_path",
        type=str,
        help="The path to the text file to read.",
    )
    args = parser.parse_args()

    try:
        with open(args.file_path, "r", encoding="utf-8") as file:
            prompt_text = file.read()
        print("File content successfully read!")
        #print(f"Content of '{args.file_path}':\n{prompt_text}")
    except FileNotFoundError:
        print(f"Error: File '{args.file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()

    gen_outputs = model.generate(
        input_ids,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=False,
        top_k=50
    )
    generated = tokenizer.decode(gen_outputs[0])


    '''result = generated
    print(f"User: {test_sentence}\nResult:\n{result.strip()}")'''

    def extract_parse(output):
        parse_index = output.find("LLAMA2 Parse:")
        if parse_index != -1:
            parse_content = output[parse_index + len("LLAMA2 Parse:"):].strip()
            # remove repeated "Now I have ..."
            stop_index = parse_content.find("Now I have")
            if stop_index != -1:
                parse_content = parse_content[:stop_index].strip()
            return parse_content
        return output.strip()

    result = extract_parse(generated)
    #print(f"User:\n{test_sentence}\nResult:\n{result}")
    print(f"Result:\n{result}")
if __name__ == "__main__":
    main()
