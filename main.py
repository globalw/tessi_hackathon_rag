import argparse

from embeddings.loader import DocumentLoader
from rag.request_evaluator import RequestEvaluator, RequestEvaluationResult, RequestBillability

from fileio.csv import load_csv_rows, write_dict_to_csv

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run main script with optional 'init' command")
    parser.add_argument("command", nargs="?", default="run", help="Optional command, e.g., 'init'")

    args = parser.parse_args()

    if args.command == "init":
        load_contracts()
    else:
        evaluator = RequestEvaluator()
        results: list[dict[str, str]] = []
        for row in load_csv_rows("./test/qa/QA_billability.csv"):
            query = row['question']
            
            try:
                response = evaluator.evaluate_customer_request(
                    request=query,
                )
            except Exception as e:
                response = RequestEvaluationResult(
                    conclusion = RequestBillability.UNKNOWN,
                    reason = f"error execution response: {e}"
                )

            results.append(dict(
                question=query,
                classification=response.conclusion,
                reason=response.reason
            ))
        
        import json
        print(json.dumps(results, indent=4))

        write_dict_to_csv(
            path = "output.csv",
            data = results,
        )