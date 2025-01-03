import torch
from transformers import LlamaForCausalLM, AutoTokenizer

def main():
    # 模型名称
    model_name = "meta-llama/Llama-2-7b-hf"

    # 加载模型和tokenizer
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

    test_sentence = [

        "I'm looking for chicken recipes"

    ]

    # 提供示例（简单形式）
    '''examples = [
        {"input": "The cat sat on the mat.", "parse": "(s / sit-01 :ARG0 (c / cat) :location (m / mat))"},
        {"input": "John loves Mary.", "parse": "(l / love-01 :ARG0 (p / person :name (n / name :op1 'John')) :ARG1 (p2 / person :name (n2 / name :op1 'Mary')))"},
        {"input": "She is reading a book.", "parse": "(r / read-01 :ARG0 (p / person :gender female) :ARG1 (b / book))"},
        {"input": "The dog barked loudly.", "parse": "(b / bark-01 :ARG0 (d / dog) :manner (l / loud))"},
        {"input": "Alice went to the park.", "parse": "(g / go-01 :ARG0 (p / person :name (n / name :op1 'Alice')) :destination (p2 / park))"},
        {"input": "The bird sang a beautiful song.", "parse": "(s / sing-01 :ARG0 (b / bird) :ARG1 (s2 / song :mod (b2 / beautiful)))"},
        {"input": "The boy kicked the ball.", "parse": "(k / kick-01 :ARG0 (b / boy) :ARG1 (b2 / ball))"},
        {"input": "The sun is shining brightly.", "parse": "(s / shine-01 :ARG0 (s2 / sun) :manner (b / bright))"},
        {"input": "The girl is drawing a picture.", "parse": "(d / draw-01 :ARG0 (g / girl) :ARG1 (p / picture))"},
        {"input": "He wrote a letter to his friend.", "parse": "(w / write-01 :ARG0 (h / he) :ARG1 (l / letter) :ARG2 (f / friend :poss h))"},
        
    ]'''

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
        """
        构造包含多个示例的 prompt
        """
        prompt = "Below are examples of converting user utterances into Mtop semantic parses:\n"
        for i, ex in enumerate(examples, start=1):
            prompt += f"\nExample {i}:\nUser: {ex['input']}\nParse: {ex['parse']}\n"
        prompt += "\nNow I have a new user utterance.\n"
        prompt += f"User: {new_utterance}\n"
        prompt += "LLAMA2 Parse:"
        return prompt

    # 构造 prompt
    prompt_text = build_prompt(examples, test_sentence)
    #print(f"\n=== Prompt ===\n{prompt_text}\n")

    # 编码
    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()

    # 推理
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
        """
        提取生成结果中 Parse 后的部分，并停止在第一个完整结果后
        """
        parse_index = output.find("LLAMA2 Parse:")
        if parse_index != -1:
            parse_content = output[parse_index + len("LLAMA2 Parse:"):].strip()
            # 检查多余的 "Now I have" 或类似的重复内容
            stop_index = parse_content.find("Now I have")
            if stop_index != -1:
                parse_content = parse_content[:stop_index].strip()
            return parse_content
        return output.strip()

    result = extract_parse(generated)
    print(f"User:\n{test_sentence}\nResult:\n{result}")

if __name__ == "__main__":
    main()
