from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama

from pydantic import BaseModel
from enum import Enum

from vector_db.db import get_vector_db

class RequestBillability(str, Enum):
    BILLABLE = "covered"
    UNBILLABLE = "not covered"
    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value):
        return cls.UNKNOWN

class RequestEvaluationResult(BaseModel):
    conclusion: RequestBillability
    reason: str

class RequestEvaluator:
    def __init__(self):
        self._rag_prompt = ChatPromptTemplate.from_template(
            """You are an assistant to evaluate whether or not answering a certain customer request is covered in a service contract or is not covered and can be billed extra.
Use the following pieces of retrieved context to answer the question.

    `Conclusion` is either "covered", "not covered", or "unknown", and `reason` is a short explanation why you made the decision.

If the contract covers the request, set `conclusion` to "covered", if not, set it to "not covered". If you don't know the answer, just set it to "unknown". Only reply in a single word.

<context>
{context}
</context>

Determine if the following customer request is "covered" or "not covered". If you don't know, it's "unknown":

<question>
{question}
</question>

{format_instructions}
"""
        )

        self._model = ChatOllama(
            model="llama3.2"
        )
    
    def _format_docs(self, docs: list[Document]):
        """
        Concat retrieved documents into string and discard metadata
        """
        return "\n\n".join(f"""
                        From {doc.metadata.get("source")}:
                        {doc.page_content}
        """ for doc in docs)
    
    def evaluate_customer_request(self, request: str) -> RequestEvaluationResult:
        vdb = get_vector_db()
        retriever = vdb.as_retriever(# automatically retrieve for now -- might wanna replace this with custom similarity ranking
            kwargs=dict(
                k=4
            )
        )

        structured_model = self._model.with_structured_output(
            schema = RequestEvaluationResult,
        )

        parser = PydanticOutputParser(pydantic_object=RequestEvaluationResult)

        prompt = self._rag_prompt.partial(
            format_instructions = parser.get_format_instructions()
        )

        chain = (
            {"context": retriever | self._format_docs, "question": RunnablePassthrough()}
            | prompt
            | self._model
            | parser
        )

        # Run
        res: RequestEvaluationResult = chain.invoke(request)
        return res