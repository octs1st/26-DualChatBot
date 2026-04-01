from openai import OpenAI
from langchain_community.document_loaders import TextLoader
# PDF 불러올 경우 import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

file_path = r"C:\Users\rocom\OneDrive\문서\26-1\reference\p2p.txt"
loader = TextLoader(file_path,encoding='utf-8')
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
splits = text_splitter.split_documents(data)


print(data)
