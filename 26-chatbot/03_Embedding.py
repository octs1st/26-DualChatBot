from langchain_core.embeddings import Embeddings
from openai import OpenAI
from typing import List

class MyEmbeddings(Embeddings):
    def __init__(self, base_url, api_key='lm-studio'):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
    
    def embed_documents(self, texts: List[str], model='text-embedding-nomic-embed-text-v1.5') -> List[List[float]]:
        texts = list(map(lambda text:text.replace("\n", ' '), texts))
        datas = self.client.embeddings.create(input=texts, model='text-embedding-nomic-embed-text-v1.5').data
        return list(map(lambda data:data.embedding, datas))
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
    
emb_model = MyEmbeddings(base_url="http://127.0.0.1:1234/v1")   
emb_vectors = emb_model.embed_documents([
    "안녕하세요.",
    '잘부탁합니다.',
    '프로젝트를하자.'
])

print(emb_vectors[0][:4])
print(emb_vectors[1][:4])
print(emb_vectors[2][:4])