from llama_index.llms.llama_api import LlamaAPI
from llama_index.llms.openai import OpenAI
import os
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader

import pandas as pd


class RAG:

    class Parameters(BaseModel):
        LLAMAINDEX_OLLAMA_BASE_URL: str
        LLAMAINDEX_MODEL_NAME: str
        LLAMAINDEX_EMBEDDING_MODEL_NAME: str

    def __init__(self):
        self.documents = None
        self.index = None

        self.parameters = self.Parameters(
            LLAMAINDEX_OLLAMA_BASE_URL=os.getenv(
                "LLAMAINDEX_OLLAMA_BASE_URL", "http://localhost:11434"),
            LLAMAINDEX_MODEL_NAME=os.getenv(
                "LLAMAINDEX_MODEL_NAME", "llama3.2:1b"),
            LLAMAINDEX_EMBEDDING_MODEL_NAME=os.getenv(
                "LLAMAINDEX_EMBEDDING_MODEL_NAME", "jina/jina-embeddings-v2-base-en:latest"),
        )

    def on_startup(self):
        Settings.embed_model = OllamaEmbedding(
            model_name=self.parameters.LLAMAINDEX_EMBEDDING_MODEL_NAME,
            base_url=self.parameters.LLAMAINDEX_OLLAMA_BASE_URL,
        )
        Settings.llm = Ollama(
            model=self.parameters.LLAMAINDEX_MODEL_NAME,
            base_url=self.parameters.LLAMAINDEX_OLLAMA_BASE_URL,
            request_timeout=3600.0,
        )
        # Settings.llm = LlamaAPI(model='llama3.2-3b', api_key=LLM_KEY)

        self.documents = SimpleDirectoryReader("test/dataset").load_data()
        self.index = VectorStoreIndex.from_documents(self.documents)

    def on_shutdown(self):
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        for _ in range(9):
            try:
                chat_engine = self.index.as_chat_engine()

                question1 = f'You are a company that signed a contract with a client. Find if the following query is included in the service provided by the company. Query: {query}'
                explanation_response = chat_engine.chat(question1).response

                question2 = f'Based on {explanation_response}, would you say this query is billable, query : {query} ? Answer in one word by "yes" or "no".'
                in_service_response = chat_engine.chat(question2).response

                if in_service_response.lower() not in ['yes', 'no']:
                    raise (TypeError('Not the expected output'))

            except:
                print('Meh...')

        return f'{in_service_response}&&&{explanation_response}'


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


if __name__ == "__main__":
    df = pd.read_csv('test/qa/QA_billability.csv')
    res = []

    for query in df['question']:
        response = run_query_sync(query)
        print('\n--')
        print(f'Query: {query}')
        print(f"Response: {response}")
        res.append({
            'question': query,
            'expected_answer': 'Not Billable' if response.split('&&&')[0].lower() == 'no' else 'Billable',
            'context': response.split('&&&')[1],
            'debug_answer': response.split('&&&')[0]})

    pd.DataFrame(res).to_csv('test/qa/QA.csv')
