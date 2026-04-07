from openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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
                api_key="lm-studio",
                model="llama-3-korean-bllossom-8b",
                temperature=0.7
                )
embeddings = MyEmbeddings(base_url="http://127.0.0.1:1234/v1")  

completion = client.chat.completions.create(
  messages=[
    {"role": "system", "content": "문장 끝날 때 이모티콘을 써줘"},
    {"role": "user", "content": "교내 금지 대상 프로그램이 뭐야"}
  ],
)

file_path = r"C:\Users\rocom\OneDrive\문서\26-1\26-DualChatBot\reference\p2p.txt"
loader = TextLoader(file_path,encoding='utf-8')
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20,
                                               separators=["\n\n", "\n", "(?<=\. )", " ", ""], length_function=len)
splits = text_splitter.split_documents(data)

emb_vectors = embeddings.embed_documents([
    "안녕하세요.",
    '잘부탁합니다.',
    '프로젝트를하자.'
]) 

print(completion.choices[0].message.content)
