from openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

emb_model = OpenAIEmbeddings(
    model="text-embedding-nomic-embed-text-v1.5", 
    openai_api_base="http://127.0.0.1:1234/v1", 
    api_key="lm-studio"
)

file_path = r"C:\Users\rocom\OneDrive\문서\26-1\26-DualChatBot\reference\p2p.txt"
loader = TextLoader(file_path,encoding='utf-8')
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
splits = text_splitter.split_documents(data)

vector_store = Chroma.from_documents(
    documents=splits, 
    embedding=emb_model,
    persist_directory = './VectorDB')
retriever = vector_store.as_retriever()

print(splits)
