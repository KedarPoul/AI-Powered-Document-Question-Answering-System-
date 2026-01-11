from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


def clean_text(text: str) -> str:
    return text.encode("utf-8", errors="ignore").decode("utf-8")


def create_vector_store(chunks):
    embeddings = OllamaEmbeddings(model="mistral")

    clean_chunks = [clean_text(chunk) for chunk in chunks]

    vectorstore = FAISS.from_texts(clean_chunks, embeddings)
    return vectorstore
