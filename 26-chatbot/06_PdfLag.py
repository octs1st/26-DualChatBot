from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List

class MyEmbeddings(Embeddings):
    def __init__ (self, base_url, api_key='lm-studio'):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
    def embed_documents(self, texts: List[str], model='bge-m3') -> List[List[float]]:
        texts = list(map(lambda text:text.replace("\n", ' '), texts))
        datas = self.client.embeddings.create(input=texts, model=model).data
        return list(map(lambda data:data.embedding, datas))
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
    

file_path = './files/bylaws.pdf'
client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
embeddings = MyEmbeddings(base_url="http://127.0.0.1:1234/v1")  
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, 
                                               chunk_overlap=20,
                                               separators=["\n\n", "\n", r"(?<=\. )", " ", ""], 
                                               length_function=len)

def embed_file(path):
    ext = path.split('.')[-1].lower()
    Loader = {'txt':TextLoader, 'pdf':PyPDFLoader}[ext]
    loader = Loader(path)
    docs = loader.load_and_split(text_splitter=text_splitter)
    
    vector_store = Chroma.from_documents(
        documents = docs,
        embedding = embeddings,
        persist_directory = './VectorDB',)
    retriever = vector_store.as_retriever()
    return retriever
retriever = embed_file(file_path)

query = "동서대학교의 학생대표 자격은 어떻게 돼?"
rel_docs = retriever.invoke(query)
context = "\n".join([doc.page_content for doc in rel_docs])

completion = client.chat.completions.create(
    model="llama-3.2-Korean-Bllossom-3B",
    temperature=0.7,
    messages=[
        {"role": "system", "content": f"학생들이 이해하기 쉽게 친절하게 설명해줘. 지식은 내가 준 pdf에서만 가져와,{context}"},
        {"role": "user", "content": query}
  ],
)

print(completion.choices[0].message.content)