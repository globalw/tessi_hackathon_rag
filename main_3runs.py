import os
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
import pandas as pd
import json
import random  # Importing the random module
from collections import Counter

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
        self, user_message: str, model_id: str, messages: List[dict], body: dict, top_k: int = 5
    ) -> Union[str, Generator, Iterator]:
        query_engine = self.index.as_query_engine(streaming=True, top_k=top_k)
        response = query_engine.query(user_message)
        return response.response_gen


def run_query_sync(query: str, top_k: int = 5) -> str:
    rag = RAG()
    rag.on_startup()
    
    user_message = query
    model_id = "llama3.2:3b"
    messages = [{"role": "user", "content": user_message}]
    body = {}
    
    # return top_k vectors (and not the default)
    response_gen = rag.pipe(user_message, model_id, messages, body, top_k=top_k)
    response_text = ''.join(response_gen)

    rag.on_shutdown()
    return response_text

def get_most_frequent_class(results):
    return Counter(results).most_common(1)[0][0]

if __name__ == "__main__":

    prompt = f"""
    Classify the following request among the following request categories:
        - Emergency Maintenance (IMBL Scanner Breakdown - Immediate Repair Required)
        - Routine Maintenance (Scheduled Maintenance Request for IMBL Scanner)
        - Upgrades (Request for Software Upgrade on IMBL Scanner)
        - Training (Request for Additional IMBL Scanner Training)
        - Replacement Under Warranty (Request for Replacement of Defective IMBL Scanner Part)
        - Customization (Request for Customization of IMBL Scanner Settings)
        - User-Caused Damage (Repair Request for IMBL Scanner - Accidental Damage)
        - New Projects (Request for Installation and Setup of New IMBL Scanner; Request for Remote Troubleshooting - IMBL Scanner Software Issue)
        - Performance Reports (Request for Detailed Monthly Performance Report)
        - Other (all other requests)

    Then check if the following request is covered under the Service Level Agreement (SLA) or not. Return the relevant passages from the SLA under "context" below.
    
    Then decide if the request is covered by the SLA, in that case the request is Billable, else the request is Not Billable. Also consider the Billable / Not Billable information depending on the request category.

    Return your reply in the following JSON format, nothing else, and follow strictly this format:
    {{
        category: [one of the request category above],
        context: [relevant passage from the Service Level Agreement, do not include the customer request here],
        billable: [Billable / Not Billable]
    }}
    """

    df_questions = pd.read_csv('test/qa/QA_billability.csv')
    df_questions['predicted_category'] = None
    df_questions['predicted_context'] = None
    df_questions['predicted_answer'] = None

    # Initialize RAG once
    rag = RAG()
    rag.on_startup()

    for case_nb in range(df_questions.shape[0]):
        question = df_questions['question'][case_nb]
        responses = []
        print("\n================================")
        print(f'case_nb', case_nb)
        print(f'question:', question)
        
        for run in range(3):
            while True:
                try:
                    seed = random.randint(0, 10000)  # Change the seed for each run
                    query = prompt + question + f"\n\nRandom Seed: {seed}"  # Adding seed to enforce variation
                    response = run_query_sync(query, top_k=5)
                    response_json = json.loads(response)
                    responses.append(response_json["billable"])
                    print(f"Run {run + 1} response with seed {seed}: {response_json}")
                    break
                except json.JSONDecodeError:
                    print("JSONDecodeError encountered. Retrying...")

        most_likely_class = get_most_frequent_class(responses)
        df_questions.loc[case_nb, 'predicted_answer'] = most_likely_class
        print(f'Most likely class for case {case_nb}: {most_likely_class}')
        
        print(f'expected_answer:', df_questions['expected_answer'][case_nb])
        print(f'expected_context:', df_questions['context'][case_nb])
        
        if most_likely_class == df_questions['expected_answer'][case_nb]:
            print('Prediction correct')
        else:
            print('PREDICTION WRONG !!!')
        print("--------------------------------")

    rag.on_shutdown()

print("Processing completed.")



df_csv = df_questions[['question', 'predicted_answer', 'predicted_context']]
df_csv.columns =['question', 'answer', 'context']
df_csv.to_csv('billability_submission_3runs.csv')
df_csv