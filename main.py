import argparse

from embeddings.loader import DocumentLoader
from rag.request_evaluator import RequestEvaluator

from dotenv import load_dotenv
import os

load_dotenv()

if(os.getenv("DEBUG") == "true"):
    import langchain
    langchain.debug = True

def load_contracts():
    print("Loading contracts...")
    dl = DocumentLoader()
    
    dl.load_directory(
        path="./test/dataset/",  # Txt files
        glob="**/*.txt", # loads all .txt files in ./test/dataset/ and all subdirectories (recursively)
    )

def eval_request(request: str) -> str:
    evaluator = RequestEvaluator()
    return evaluator.evaluate_customer_request(
        request=request,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run main script with optional 'init' command")
    parser.add_argument("command", nargs="?", default="run", help="Optional command, e.g., 'init'")

    args = parser.parse_args()

    if args.command == "init":
        load_contracts()
    else:
        # Default behavior
        query = "Subject: IMBL Scanner Breakdown - Immediate Repair Required Dear Support Team, We are experiencing a sudden breakdown of our IMBL Scanner, rendering it non-operational. We request immediate assistance for emergency repairs to restore functionality as soon as possible. Thank you for your prompt attention to this matter. Best regards, [Customer Name]"
        response = eval_request(query)
        print(f"\nResponse: {response}")
