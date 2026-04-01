from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")

completion = client.chat.completions.create(
  model="llama-3-korean-bllossom-8b",
  messages=[
    {"role": "system", "content": "졸업 작품 테스트"},
    {"role": "user", "content": "나한테 인사해"}
  ],
  temperature=0.7,
)

print(completion.choices[0].message.content)