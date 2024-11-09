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

    def ask(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        response_gen = self.pipe(user_message, model_id, messages, body)
        response_text = ''.join(response_gen)
        return response_text

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        query_engine = self.index.as_query_engine(streaming=True)
        # chat_engine = self.index.as_chat_engine(streaming=True)
        response = query_engine.query(user_message)
        print(response)
        return response.response_gen


def run_query_sync(query: str, handler = None, expectedAnswer:str = '', expectedContext:str = '') -> str:

    if handler is None:
        # No handler passed -> use default
        handler = handlerReask

    # Evaluate question
    answerObject = evaluate(handler, query, expectedAnswer, expectedContext)
    # Print answer
    printAnswer(answerObject)

    response_text = f"{answerObject.question}, {'Billable' if answerObject.billable else 'Not Billable'}, {answerObject.context}"

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
            run_query_sync(query, handler, expectedAnswer, expectedContext)

            # answerObject = evaluate(handler, query, expectedAnswer, expectedContext)
            # printAnswer(answerObject)
        # Ask Question

        # Billable
        break
    return

def evaluate( handler, question, expectedAnswer, expectedContext ):
    answerObject = handler(question, expectedAnswer, expectedContext)
    return answerObject

def printAnswer( answerObject: answ ):
    print(f"\n------- {answerObject.handler} -------")
    print(f"Question: {answerObject.question}")
    print(f"Response: {answerObject.answer}")
    print(f"Billable: {'Billable' if answerObject.billable else 'Not Billable'}")
    print(f"Context: {answerObject.context}")
    print(f"Expected answer: {answerObject.expectedAnswer}")
    print(f"Expected context: {answerObject.expectedContext}")
    print(f"---------------------------")

    return

def handlerDummy(question, expectedAnswer, expectedContext) -> answ:
    handler = 'handlerDummy'
    answer = 'No answer'
    billable = True
    context = 'We bill everything.'
    answerObject = answ(handler, question, answer, billable, context, expectedAnswer, expectedContext)
    return answerObject

def handlerReask(question, expectedAnswer, expectedContext) -> answ:
    # Instancate object
    rag = RAG()
    rag.on_startup()

    # Prepare values
    user_message = question
    model_id = "llama3.2:3b"
    messages = [{"role": "user", "content": user_message}]
    body = {}

    answer = rag.ask(question, model_id, messages, body)
    handler = 'handlerReask'
    billableAnswer =  rag.ask(f'Say "Yes" or "No": Is "{answer}" part of the contract in the vectorstore', model_id, messages, body)
    billableAnswer2 =  rag.ask(f'Is "{answer}" billable accoring the vectorstore', model_id, messages, body)
    exclusions =  rag.ask(f'Is "{question}" handled part of the exclusions at the vectorstore.', model_id, messages, body)
    # print(billableAnswer)
    # print(billableAnswer2)
    isExcluded = 'yes' in exclusions.lower()
    isBillable = 'yes' in billableAnswer.lower() or 'unclear' in billableAnswer2.lower()
    if isBillable and not isExcluded:
        billable = True
        context = rag.ask(f'Where is "{answer}" included at the contract in the vectorstore', model_id, messages, body)
    else:
        billable = False
        if isExcluded:
            context = rag.ask(f'Why is "{answer}" not billable at the contract in the vectorstore', model_id, messages, body)
        else:
            context = rag.ask(f'Where is "{question}" excluded at the contract in the vectorstore', model_id, messages, body)

    answerObject = answ(handler, question, answer, billable, context, expectedAnswer, expectedContext)

    rag.on_shutdown()

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
