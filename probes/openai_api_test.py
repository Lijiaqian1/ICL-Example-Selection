import openai
import os


API_KEY = os.getenv("OPENAI_API_KEY")  # 读取环境变量
client = openai.OpenAI(api_key=API_KEY)


try:
    print("Testing API key...")
    models = client.models.list()
    print("API Key is valid. Available models:", models)
except Exception as e:
    print("API Key might be invalid:", e)


# 1. 创建 OpenAI 客户端
client = openai.OpenAI(api_key=API_KEY)

# 2. 发送请求
def test_openai_api():
    try:
        print("Sending request to OpenAI API...")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! How are you today?"}
            ],
            temperature=0.7,
            max_tokens=50
        )

        print("OpenAI API response:")
        print(response.choices[0].message.content)

    except Exception as e:
        print("API request failed:", str(e))

if __name__ == "__main__":
    test_openai_api()
