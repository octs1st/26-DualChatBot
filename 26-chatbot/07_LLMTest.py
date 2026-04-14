from transformers import AutoConfig
from sentence_transformers import SentenceTransformer

model = "Bllossom/llama-3.2-Korean-Bllossom-3B" 
emb_model = 'BAAI/bge-m3'
config = AutoConfig.from_pretrained(model)
emb_test = SentenceTransformer(emb_model, trust_remote_code=True)
dimension = emb_test.get_sentence_embedding_dimension()

print(f"Hidden Size: {config.hidden_size}")
print(f"Number of Layers: {config.num_hidden_layers}")
print(f"Number of Heads: {config.num_attention_heads}")
print(f"Vocabulary Size: {config.vocab_size}")
print(f"{dimension}")

#이전에는 nomic 임베딩 모델 사용했지만 차원이 많이 차이나서 변경