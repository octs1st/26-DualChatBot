from langchain_community.document_loaders import UnstructuredURLLoader
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


urls = [
    "https://uni.dongseo.ac.kr/sw/index.php?pCode=MN1000093"
]

loader = UnstructuredURLLoader(urls=urls)
docs = loader.load() 

print(len(urls))