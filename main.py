import argparse

from embeddings.chunking import DocumentLoader
from langchain_community.document_loaders import DirectoryLoader # TODO move into 
from langchain_community.document_loaders import TextLoader # enough for txt files

def load_contracts():
    print("Loading contracts...")
    dl = DocumentLoader()

    loader = DirectoryLoader(
        "./test/dataset/", # Txt files
        glob="**/*.txt", # loads all .txt files in ./test/dataset/ and all subdirectories (recursively)
        use_multithreading=True,
        loader_cls=TextLoader, # only processing text for now, so this is enough
    )
    docs = loader.load()

    for doc in docs:
        print("-------")
        print(doc)

    print("===================")
    print("Chunking")
    chunked_docs = dl.chunk(docs)

    for doc in chunked_docs:
        print("-------")
        print(doc)

    # TODO: now add these to vector DB

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run main script with optional 'init' command")
    parser.add_argument("command", nargs="?", default="run", help="Optional command, e.g., 'init'")

    args = parser.parse_args()

    if args.command == "init":
        load_contracts()
    else:
        # Default behavior
        query = "Subject: IMBL Scanner Breakdown - Immediate Repair Required Dear Support Team, We are experiencing a sudden breakdown of our IMBL Scanner, rendering it non-operational. We request immediate assistance for emergency repairs to restore functionality as soon as possible. Thank you for your prompt attention to this matter. Best regards, [Customer Name]"
        # response = run_query_sync(query)
        # print("\nResponse:")
        # print(response)
