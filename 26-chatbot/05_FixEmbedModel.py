from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List

class MyEmbeddings(Embeddings):
    def __init__ (self, base_url, api_key='lm-studio'):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
    def embed_documents(self, texts: List[str], model='text-embedding-nomic-embed-text-v1.5') -> List[List[float]]:
        texts = list(map(lambda text:text.replace("\n", ' '), texts))
        datas = self.client.embeddings.create(input=texts, model=model).data
        return list(map(lambda data:data.embedding, datas))
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
    

client = OpenAI(base_url="http://127.0.0.1:1234/v1", 
                api_key="lm-studio"
                )
embeddings = MyEmbeddings(base_url="http://127.0.0.1:1234/v1")  

completion = client.chat.completions.create(
    model="llama-3-korean-bllossom-8b",
    temperature=0.7,
    messages=[
        {"role": "system", "content": "ë¬¸ì¥ ??‚  ?•Œ ?´ëª¨í‹°ì½˜ì„ ?¨ì¤?"},
        {"role": "user", "content": "êµë‚´ ê¸ˆì?? ????ƒ ?”„ë¡œê·¸?¨?´ ë­ì•¼"}
  ],
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20,
                                               separators=["\n\n", "\n", "(?<=\. )", " ", ""], length_function=len)


def embed_file(file):
    with open('./files/p2p.txt') as f:
        f.write(file.read())
    Loader = {'txt':TextLoader, 'pdf':PyPDFLoader}[file.name.split('.')[-1].lower()]
    docs = Loader('./files/p2p.txt').load_and_split(text_splitter=text_splitter)
    splits = text_splitter.split_documents(docs)
    vector_store = Chroma.from_documents(
        documents = docs,
        embedding = embeddings,
        persist_directory = './VectorDB',)
    retriever = vector_store.as_retriever()
    return retriever
        

print(completion.choices[0].message.content)
