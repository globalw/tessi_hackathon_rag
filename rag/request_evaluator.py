from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import ChatOllama

from vector_db.db import get_vector_db

class RequestEvaluator:
    def __init__(self):
        self._rag_prompt = ChatPromptTemplate.from_template(
            """You are an assistant to evaluate whether or not answering a certain customer request is included in a service contract or can be billed extra.
Use the following pieces of retrieved context to answer the question.

If the contract covers the request, say "included", if not, say "billable". If you don't know the answer, just say "unknown". Only reply in a single word.

<context>
{context}
</context>

Determine if the following customer request is "included" or "billable". If you don't know, say "unknown":

{question}
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
    
    def evaluate_customer_request(self, request: str) -> str:
        vdb = get_vector_db()
        retriever = vdb.as_retriever(# automatically retrieve for now -- might wanna replace this with custom similarity ranking
            kwargs=dict(
                k=4
            )
        )

        chain = (
            {"context": retriever | self._format_docs, "question": RunnablePassthrough()}
            | self._rag_prompt
            | self._model
            | StrOutputParser()
        )

        # Run
        res = chain.invoke(request)
        return res