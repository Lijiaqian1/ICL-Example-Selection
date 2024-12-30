from transformers import T5Tokenizer, T5ForConditionalGeneration

# 加载模型和 tokenizer
model_name = "t5-base"  # 或者选择更大的模型如 t5-large
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 示例输入句子
sentence = "Establishing Models in Industrial Innovation"
prompt = f"Convert the following sentence into an AMR graph: '{sentence}'"

# 编码输入
inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")  # 将输入移至 GPU

# 模型生成
outputs = model.generate(inputs, max_length=200)
generated_amr = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated AMR Graph:")
print(generated_amr)
