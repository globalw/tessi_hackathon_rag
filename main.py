import os
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader


class RAG:

    class Parameters(BaseModel):
        LLAMAINDEX_OLLAMA_BASE_URL: str
        LLAMAINDEX_MODEL_NAME: str
        LLAMAINDEX_EMBEDDING_MODEL_NAME: str

    def __init__(self):
        self.documents = None
        self.index = None

        self.parameters = self.Parameters(
            LLAMAINDEX_OLLAMA_BASE_URL=os.getenv("LLAMAINDEX_OLLAMA_BASE_URL", "http://localhost:11434"),
            LLAMAINDEX_MODEL_NAME=os.getenv("LLAMAINDEX_MODEL_NAME", "llama3.2:3b"),
            LLAMAINDEX_EMBEDDING_MODEL_NAME=os.getenv("LLAMAINDEX_EMBEDDING_MODEL_NAME", "jina/jina-embeddings-v2-base-en:latest"),
        )

    def on_startup(self):
        Settings.embed_model = OllamaEmbedding(
            model_name=self.parameters.LLAMAINDEX_EMBEDDING_MODEL_NAME,
            base_url=self.parameters.LLAMAINDEX_OLLAMA_BASE_URL,
        )
        Settings.llm = Ollama(
            model=self.parameters.LLAMAINDEX_MODEL_NAME,
            base_url=self.parameters.LLAMAINDEX_OLLAMA_BASE_URL,
        )

        self.documents = SimpleDirectoryReader("test/dataset").load_data()
        self.index = VectorStoreIndex.from_documents(self.documents)



    def on_shutdown(self):
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        query_engine = self.index.as_query_engine(streaming=True)
        response = query_engine.query(user_message)
        return response.response_gen


def run_query_sync(query: str) -> str:
    rag = RAG()
    rag.on_startup()

    user_message = query
    model_id = "llama3.2:3b"
    messages = [{"role": "user", "content": user_message}]
    body = {}

    response_gen = rag.pipe(user_message, model_id, messages, body)
    response_text = ''.join(response_gen)

    rag.on_shutdown()
    return response_text



import csv
import os

if __name__ == "__main__":
    query = "Subject: Request for Software Upgrade on IMBL Scanner Dear Support Team, We would like to upgrade our IMBL Scanner software to the latest version available. Kindly provide details on the upgrade process and any associated costs. Looking forward to your response. Best regards,"

    # Erste Abfrage
    firstResponse = run_query_sync(query)

    # Billable-Abfrage
    billableQuery = f"Evaluate if the following query is billable or non-billable. Answer with unknown if it isn't a real problem or it doesn't fit into either billable or non-billable. Only give one word as an answer (BILLABLE, NON-BILLABLE, or UNKNOWN): {query}"
    secondResponse = run_query_sync(billableQuery)

    # Kontext-Abfrage
    reasoning = "Provide the exact article number, from which you based your decision for the billability on and the article's content"
    thirdResponse = run_query_sync(reasoning)

    # Ausgabe des Billable-Status
    print("\nSecond Response (Billable Status):")
    print(secondResponse)

    # Speichere query, response und Kontext in einer CSV-Datei
    file_exists = os.path.isfile("responses.csv")
    with open("responses.csv", mode="a", newline="") as responsesFile:
        fieldnames = ["query", "firstResponse", "expected_answer", "reasoning"]
        writer = csv.DictWriter(responsesFile, fieldnames=fieldnames)

        # Schreibe die Kopfzeile nur, wenn die Datei neu erstellt wurde
        if not file_exists:
            writer.writeheader()

        # Schreibe die Daten in die entsprechenden Spalten
        writer.writerow({
            "query": query,
            "firstResponse": firstResponse,
            "expected_answer": secondResponse,
            "reasoning": thirdResponse
        })