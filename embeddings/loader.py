from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader # enough for txt files

from llmsherpa.readers import LayoutPDFReader

from config.config import settings

from vector_db.db import get_vector_db

class DocumentLoader:
    def __init__(self):
        self._text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        self._lpr = LayoutPDFReader("http://localhost:5010/api/parseDocument?renderFormat=all&useNewIndentParser=true")

    def _chunk(self, documents: list[Document]) -> list[Document]:
        # very naive chunking approach
        return self._text_splitter.split_documents(documents)
    
    def _store_chunks(self, chunked_docs: list[Document]):
        vdb = get_vector_db()
        vdb.add_documents(documents=chunked_docs) # optionally I could specify IDs via ID parameter, but won't -- updating vector store is considered out of scope

    def _chunk_and_store_docs(self, docs: list[Document]):
        chunked_docs = self._chunk(docs)

        print(f"Created {len(chunked_docs)} chunks from provided docs")

        self._store_chunks(chunked_docs)
    
    def load_directory(self, path: str, glob: str):
        loader = DirectoryLoader(
            path=path,
            glob=glob,
            use_multithreading=True,
            loader_cls=TextLoader, # only processing text for now, so this is enough
        )

        docs = loader.load()

        print(f"Loaded {len(docs)} documents")

        self._chunk_and_store_docs(docs)

    def load_file_new(self, path: str):
        parsed_doc = self._lpr.read_pdf(path)

        chunks: list[Document] = []
    
        for chunk in parsed_doc.chunks():
            if settings.DEBUG:
                print(chunk.to_context_text())
                print("\n-------\n")
            chunks.append(
                Document(
                    page_content=chunk.to_context_text(),
                    metadata={
                        "source": path
                    }
                )
            )
        
        print(f"Created {len(chunks)} chunks from {path}")
        self._store_chunks(chunks)