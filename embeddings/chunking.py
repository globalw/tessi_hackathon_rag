from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# naive chunking
class DocumentLoader:
    def __init__(self):
        self._text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

    def chunk(self, documents: list[Document]) -> list[Document]:
        return self._text_splitter.split_documents(documents)