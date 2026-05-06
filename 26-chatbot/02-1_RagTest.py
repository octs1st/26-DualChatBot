from openai import OpenAI
from langchain_community.document_loaders import TextLoader

client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")

completion = client.chat.completions.create(
  model="llama-3-korean-bllossom-8b",
  messages=[
    {"role": "system", "content": "반말로 말하고 꼭 문장 끝에는 느낌표를 붙여야 함."},
    {"role": "user", "content": "교내 금지 프로그램이 뭐야?"}
  ],
  temperature=0.7,
)

loader = TextLoader('./p2p.txt')
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
splits = text_splitter.split_documents(data)

print(completion.choices[0].message.content)