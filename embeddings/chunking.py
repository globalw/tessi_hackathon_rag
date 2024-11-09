from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader # enough for txt files

class DocumentLoader:
    def __init__(self):
        self._text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

    def _chunk(self, documents: list[Document]) -> list[Document]:
        # very naive chunking approach
        return self._text_splitter.split_documents(documents)
    
    def _chunk_and_store_docs(self, docs: list[Document]):
        chunked_docs = self.chunk(docs)
        
        # TODO: now add these to vector DB
    
    def load_directory(self, path: str, glob: str):
        loader = DirectoryLoader(
            path=path,
            glob=glob,
            use_multithreading=True,
            loader_cls=TextLoader, # only processing text for now, so this is enough
        )

        docs = loader.load()

        self._chunk_and_store_docs(docs)