from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")

completion = client.chat.completions.create(
  model="llama-3-korean-bllossom-8b",
  messages=[
    {"role": "system", "content": "문장 끝마다 줄 바꿈해."},
    {"role": "user", "content": "교내 금지 대상 프로그램이 뭐야"}
  ],
  temperature=0.7,
)

print(completion.choices[0].message.content)