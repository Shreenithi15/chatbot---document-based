from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio"
)

response = client.chat.completions.create(
    model="qwen2.5-3b-instruct:2",
    messages=[
        {
            "role": "user",
            "content": "Hello! Introduce yourself in one sentence."
        }
    ],
    temperature=0.7
)

print(response.choices[0].message.content)
