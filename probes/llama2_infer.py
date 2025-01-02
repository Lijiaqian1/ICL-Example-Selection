'''import torch
from transformers import LlamaForCausalLM, AutoTokenizer

def main():
    # 1. 模型名称(可换为本地路径)
    model_name = "meta-llama/Llama-2-7b-hf"  # 若你有自己的repo可换: "path/to/local/llama2-7b"

    # 2. 加载tokenizer & model
    print(f"Loading LLaMA2-7B from {model_name} ...")
    #tokenizer = LlamaTokenizer.from_pretrained(model_name)
    access_token = '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'
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",   # 自动分配到GPU(需要 accelerate/transformers支持)
        # load_in_8bit=True, # 如果显存不足，可以考虑量化选项
    )
    # 确保有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 准备一个或多个测试句子（semantic parsing 任务）
    # 你在后续可更换为 TreeDST / MTop(en) / SMCalflow 之类的数据
    test_sentences = [
        "Book a restaurant for me in Boston tomorrow at 7pm.",
        "Set an alarm for 6 AM and remind me to take my medication."
    ]

    # 3. 这里示范 In-Context Learning：提供示例(句 -> 解析)
    #   假设你想让它做 semantic parsing 生成意图/slots
    #   你可以在 prompt 里写“Example input -> Example parse”，这样LLM可能模仿
    examples = [
        {
            "input": "Order me an Uber to the airport at 5pm.",
            "parse": "Intent: RequestRide\nSlot: time=17:00, destination=airport"
        },
        {
            "input": "Remind me to buy groceries tonight at 8pm.",
            "parse": "Intent: CreateReminder\nSlot: time=20:00, task=buy groceries"
        }
    ]

    def build_prompt(examples, new_utterance):
        """
        将若干示例拼成prompt, 并在末尾给出新的句子, 引导LLM生成解析
        """
        prompt = "Below are examples of converting user utterances into semantic parses:\n"
        for i, ex in enumerate(examples, start=1):
            prompt += f"\nExample {i}:\nUser: {ex['input']}\nParse: {ex['parse']}\n"
        prompt += "\nNow I have a new user utterance.\n"
        prompt += f"User: {new_utterance}\n"
        prompt += "Parse:"
        return prompt

    # 4. 推理演示
    for idx, sentence in enumerate(test_sentences):
        # 构造 prompt
        prompt_text = build_prompt(examples, sentence)
        print(f"\n=== Prompt {idx+1} ===\n{prompt_text}\n")

        # 编码
        inputs = tokenizer(prompt_text, return_tensors="pt")
        input_ids = inputs["input_ids"].cuda()  # or to device

        # 生成
        gen_outputs = model.generate(
            input_ids,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=False,   # 这里用贪心 / beam=1
            top_k=50
        )
        generated = tokenizer.decode(gen_outputs[0])
        # 截取 prompt部分, 只取“Parse:”之后
        parse_index = generated.find("Parse:")
        if parse_index != -1:
            result = generated[parse_index + len("Parse:") :]
        else:
            result = generated

        print(f"User: {sentence}\nLLaMA2 parse:\n{result.strip()}")

if __name__ == "__main__":
    main()

import torch
from transformers import LlamaForCausalLM, AutoTokenizer

def main():
    # 1. 模型名称(可换为本地路径)
    model_name = "meta-llama/Llama-2-7b-hf"  # 若你有自己的repo可换: "path/to/local/llama2-7b"

    # 2. 加载tokenizer & model
    print(f"Loading LLaMA2-7B from {model_name} ...")
    #tokenizer = LlamaTokenizer.from_pretrained(model_name)
    access_token = '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'
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",   # 自动分配到GPU(需要 accelerate/transformers支持)
        # load_in_8bit=True, # 如果显存不足，可以考虑量化选项
    )
    # 确保有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 准备一个或多个测试句子（semantic parsing 任务）
    # 你在后续可更换为 TreeDST / MTop(en) / SMCalflow 之类的数据
    test_sentences = [
        "Naif Arab Academy for Security Sciences is run by an Arab Interior Ministers' Council.",
        "EBay Announces Today It will Acquire Ticket Website StubHub for 310 Million US Dollars"
    ]

    # 3. 这里示范 In-Context Learning：提供示例(句 -> 解析)
    #   假设你想让它做 semantic parsing 生成意图/slots
    #   你可以在 prompt 里写“Example input -> Example parse”，这样LLM可能模仿
    examples = [
        {
            "input": "The Riyadh-based Naif Arab Academy for Security Sciences said in a statement that it was running a two-week workshop for 50 anti-terrorism experts.",
            "parse": "(s / say-01\n   :ARG0 (u / university\n            :wiki -\n            :name (n / name\n                     :op1 \"Naif\"\n                     :op2 \"Arab\"\n                     :op3 \"Academy\"\n                     :op4 \"for\"\n                     :op5 \"Security\"\n                     :op6 \"Sciences\")\n            :ARG1-of (b / base-01\n                        :location (c / city\n                                     :wiki \"Riyadh\"\n                                     :name (n2 / name\n                                               :op1 \"Riyadh\"))))\n   :ARG1 (r / run-01\n            :ARG0 u\n            :ARG1 (w / workshop\n                     :beneficiary (p / person\n                                     :quant 50\n                                     :ARG1-of (e / expert-01\n                                                 :ARG2 (c2 / counter-01\n                                                           :ARG1 (t2 / terrorism))))\n                     :duration (t / temporal-quantity\n                                  :quant 2\n                                  :unit (w2 / week))))\n   :medium (s2 / statement))"
        },
        {
            "input": "Naif Arab Academy for Security Sciences is based in Riyadh.",
            "parse": "(b / base-01\n   :ARG1 (u / university\n            :wiki -\n            :name (n / name\n                     :op1 \"Naif\"\n                     :op2 \"Arab\"\n                     :op3 \"Academy\"\n                     :op4 \"for\"\n                     :op5 \"Security\"\n                     :op6 \"Sciences\"))\n   :location (c / city\n                :wiki \"Riyadh\"\n                :name (n2 / name\n                          :op1 \"Riyadh\")))"
        }
    ]

    def build_prompt(examples, new_utterance):
        """
        将若干示例拼成prompt, 并在末尾给出新的句子, 引导LLM生成解析
        """
        prompt = "Below are examples of converting user utterances into abstract meaning representation (amr) semantic parses:\n"
        for i, ex in enumerate(examples, start=1):
            prompt += f"\nExample {i}:\nUser: {ex['input']}\nParse: {ex['parse']}\n"
        prompt += "\nNow I have a new user utterance.\n"
        prompt += f"User: {new_utterance}\n"
        prompt += "Parse:"
        return prompt

    # 4. 推理演示
    for idx, sentence in enumerate(test_sentences):
        # 构造 prompt
        prompt_text = build_prompt(examples, sentence)
        print(f"\n=== Prompt {idx+1} ===\n{prompt_text}\n")

        # 编码
        inputs = tokenizer(prompt_text, return_tensors="pt")
        input_ids = inputs["input_ids"].cuda()  # or to device

        # 生成
        gen_outputs = model.generate(
            input_ids,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=False,   # 这里用贪心 / beam=1
            top_k=50
        )
        generated = tokenizer.decode(gen_outputs[0])
        # 截取 prompt部分, 只取“Parse:”之后
        parse_index = generated.find("Parse:")
        if parse_index != -1:
            result = generated[parse_index + len("Parse:") :]
        else:
            result = generated

        print(f"User: {sentence}\nLLaMA2 parse:\n{result.strip()}")

if __name__ == "__main__":
    main()
'''

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
        #"The company announced a new product launch.",
        #"The bird flew over the mountains.",
        "Who is Shirley's mother"
        #"The child is playing with a toy.",
        #"The train arrived at the station on time."
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
        prompt = "Below are examples of converting user utterances into simple AMR semantic parses:\n"
        for i, ex in enumerate(examples, start=1):
            prompt += f"\nExample {i}:\nUser: {ex['input']}\nParse: {ex['parse']}\n"
        prompt += "\nNow I have a new user utterance.\n"
        prompt += f"User: {new_utterance}\n"
        prompt += "Parse:"
        return prompt

    # 构造 prompt
    prompt_text = build_prompt(examples, test_sentence)
    print(f"\n=== Prompt ===\n{prompt_text}\n")

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

    # 提取 Parse 部分
    parse_index = generated.find("Parse:")
    if parse_index != -1:
        result = generated[parse_index + len("Parse:") :]
    else:
        result = generated

    print(f"User: {test_sentence}\nLLaMA2 parse:\n{result.strip()}")

if __name__ == "__main__":
    main()
