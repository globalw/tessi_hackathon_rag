import os
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
import pandas as pd
from answer import Answer as answ


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
        # Tokanizer: What am i eating -> '1212 4343 2342'
        # Embedding: {1212: [xx,xx,x] 4343: [xx,xx,x] 2342: [xx,xx,x]}
        Settings.embed_model = OllamaEmbedding(
            model_name=self.parameters.LLAMAINDEX_EMBEDDING_MODEL_NAME,
            base_url=self.parameters.LLAMAINDEX_OLLAMA_BASE_URL,
        )

        # Aggregate the tokens to give them sense
        Settings.llm = Ollama(
            model=self.parameters.LLAMAINDEX_MODEL_NAME,
            base_url=self.parameters.LLAMAINDEX_OLLAMA_BASE_URL,
        )

        # Load the files in the directory
        self.documents = SimpleDirectoryReader("test/dataset").load_data()

        # Load documents into a vector storage
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

def handleQueries() -> str:
    # Read csv
    QaBillability = pd.read_csv('test/qa/QA_billability.csv')
    handlerList = [handlerDummy, handlerReask]

    # Itterate questions
    for idx, query in enumerate(QaBillability["question"]):
        expectedAnswer = QaBillability["expected_answer"][idx]
        expectedContext = QaBillability["context"][idx]
        for handler in handlerList:
            answerObject = evaluate(handler, query, expectedAnswer, expectedContext)
            printAnswer(answerObject)
        # Ask Question

        # Billable
        # break
    return

def evaluate( handler, question, expectedAnswer, expectedContext ):
    answerObject = handler(question, expectedAnswer, expectedContext)
    return answerObject

def printAnswer( answerObject: answ ):
    print(f"\n------- {answerObject.handler} -------")
    print(f"Question: {answerObject.question}")
    print(f"Response: {answerObject.answer}")
    print(f"Billable: {answerObject.billable}")
    print(f"Context: {answerObject.context}")
    print(f"Expected answer: {answerObject.expectedAnswer}")
    print(f"Expected context: {answerObject.expectedContext}")
    print(f"---------------------------")

    return

def handlerDummy(question, expectedAnswer, expectedContext) -> answ:
    handler = 'handlerDummy'
    answer = 'No answer'
    billable = 'Yes'
    context = 'We bill everything.'
    answerObject = answ(handler, question, answer, billable, context, expectedAnswer, expectedContext)
    return answerObject

def handlerReask(question, expectedAnswer, expectedContext) -> answ:
    answer = run_query_sync(question)
    handler = 'handlerReask'
    billableAnswer = run_query_sync(f'Is "{answer}" part of the contract in the vectorstore')
    print(billableAnswer)
    if 'yes' in billableAnswer.lower():
        billable = 'Yes'
        context = run_query_sync(f'Where is "{answer}" included at the contract in the vectorstore')
    else:
        billable = 'No'
        context = 'The service is not part of the contract.'
    answerObject = answ(handler, question, answer, billable, context, expectedAnswer, expectedContext)
    return answerObject

if __name__ == "__main__":

    # cls.qa_data = pd.read_csv('qa/QA_billability.csv')
    # query = "Subject: IMBL Scanner Breakdown - Immediate Repair Required Dear Support Team, We are experiencing a sudden breakdown of our IMBL Scanner, rendering it non-operational. We request immediate assistance for emergency repairs to restore functionality as soon as possible. Thank you for your prompt attention to this matter. Best regards, [Customer Name]"
    # query = "Subject: Ich hab da ne Frage [Customer Name]"
    # response = run_query_sync(query)
    # print("\nResponse:")
    # print(response)

    # Test
    handleQueries()
