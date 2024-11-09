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


def run_query_sync(query: str, top_k: int = 5) -> str:
     rag = RAG()
     rag.on_startup()
     
     user_message = query
     model_id = "llama3.2:3b"
     messages = [{"role": "user", "content": user_message}]
     body = {}
     
     response_gen = rag.pipe(user_message, model_id, messages, body, top_k=top_k)
     response_text = ''.join(response_gen)

     rag.on_shutdown()
     return response_text

def run_query_sync_old(query: str) -> str:
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



def get_billable_answer(long_answer):
    # Split the string by lines
    lines = long_answer.split('\n')
    # Find the line that starts with 'answer:'
    for line in lines:
        if line.startswith("answer:"):
            answer = line.split(":")[1].strip()
            break
    return answer



if __name__ == "__main__":

    prompt_old = 'Classify this question if the request is under the Service Level Agreement (SLA) or not. Return both the relevant passage in the SLA and if the request is covered or not (yes / no)'
    prompt = f"""Classify if the following request is under the Service Level Agreement (SLA) or not. Return your reply in the following format:
            answer: [Billable / Not Billable]
            context: [relevant passage from the SLA]
            """

    
    df_questions = pd.read_csv('test/qa/QA_billability.csv')
    df_questions['predicted_answer'] = 999
    #print('questions')
    #print(questions)

    for case_nb in range(0,df_questions.shape[0]):
   

        #question = "Subject: IMBL Scanner Breakdown - Immediate Repair Required Dear Support Team, We are experiencing a sudden breakdown of our IMBL Scanner, rendering it non-operational. We request immediate assistance for emergency repairs to restore functionality as soon as possible. Thank you for your prompt attention to this matter. Best regards, [Customer Name]"
        question = df_questions['question'][case_nb]
        query = prompt + question
        print("\n================================")
        print(f'case_nb', case_nb)     
        print(question)
        #print(query)
        response = run_query_sync(query)
        print("\nResponse:")
        print(response)
        pred_billable = get_billable_answer(response)
        df_questions['predicted_answer'][case_nb] = pred_billable
        print(f'expected_answer:', df_questions['expected_answer'][case_nb])
        print(f'expected_context:', df_questions['context'][case_nb])
        print("\n--------------------------------")


