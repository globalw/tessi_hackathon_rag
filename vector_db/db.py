from langchain_core.vectorstores import VectorStore
from langchain_chroma import Chroma

from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="jina/jina-embeddings-v2-base-en:latest",
)

_cached_db_instance = None

def get_vector_db() -> VectorStore:
    global _cached_db_instance

    if _cached_db_instance is None:
        print("Initializing DB")
        _cached_db_instance = Chroma(
            collection_name="tessi-rag",
            embedding_function=embeddings,
            persist_directory="./chroma_langchain_db"
        )

    return _cached_db_instance