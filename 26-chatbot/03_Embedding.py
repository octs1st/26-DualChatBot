from langchain_core.embeddings import Embeddings
from openai import OpenAI
from typing import list

class MyEmbeddings(Embeddings):
    def __init__(self, base_url, api_key='lm-studio'):
        self.clien = OpenAI(base_url, api_key=api_key)
    
MyEmbeddings()