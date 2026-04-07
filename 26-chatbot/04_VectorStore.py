from openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings


client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")

emb_model = MyEmbeddings(
    model="text-embedding-nomic-embed-text-v1.5", 
    base_url="http://127.0.0.1:1234/v1", 
    api_key="lm-studio"
)

completion = client.chat.completions.create(
  model="llama-3-korean-bllossom-8b",
  messages=[
    {"role": "system", "content": "문장 끝날 때 이모티콘을 써줘"},
    {"role": "user", "content": "교내 금지 대상 프로그램이 뭐야"}
  ],
  temperature=0.7,
)

file_path = r"C:\Users\rocom\OneDrive\문서\26-1\26-DualChatBot\reference\p2p.txt"
loader = TextLoader(file_path,encoding='utf-8')
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
splits = text_splitter.split_documents(data)

texts = [doc.page_content for doc in splits]

vector_store = Chroma.from_texts(
    texts=texts,
    embedding = emb_model,
    persist_directory = './VectorDB',
    check_embedding_ctx_length=False,
    skip_empty=True)
retriever = vector_store.as_retriever()

print(completion.choices[0].message.content)
