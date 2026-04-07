from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")

completion = client.chat.completions.create(
  model="llama-3-korean-bllossom-8b",
  messages=[
    {"role": "system", "content": "문장 끝마다 이모티콘을 넣어"},
    {"role": "user", "content": "오늘의 점심 메뉴 추천은?"}
  ],
  temperature=0.7,
)

print(completion.choices[0].message.content)