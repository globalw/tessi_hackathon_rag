import argparse

from embeddings.loader import DocumentLoader
from rag.request_evaluator import RequestEvaluator, RequestEvaluationResult, RequestBillability

from fileio.csv import load_csv_rows, write_dict_to_csv

from config.config import settings

if settings.DEBUG:
    import langchain
    langchain.debug = True

def load_contracts():
    print("Loading contracts...")
    dl = DocumentLoader()
    
    # dl.load_directory(
    #     path="./test/dataset/",  # Txt files
    #     glob="**/*.txt", # loads all .txt files in ./test/dataset/ and all subdirectories (recursively)
    # )

    dl.load_file_new(
        "./dataset/bak/Maintenance_Service_Agreement.docx"
    )

    dl.load_file_new(
        "./test/dataset/Project_Setup_Contract.txt"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run main script with optional 'init' command")
    parser.add_argument("command", nargs="?", default="run", help="Optional command, e.g., 'init'")

    if(settings.DEBUG):
        print("STARTED IN VERBOSE DEBUG MODE")

    args = parser.parse_args()

    if args.command == "init":
        load_contracts()
    else:
        evaluator = RequestEvaluator()
        results: list[dict[str, str]] = []

        # overwriting this here as it should be written according to the challenge rules, but original definitions differ to make it easier for the LLM
        billability_results = {
            RequestBillability.BILLABLE: "Billable", # included in the contract
            RequestBillability.UNBILLABLE: "Not Billable", # not included in the contract
            RequestBillability.UNKNOWN: "?" # needs manual review
        }

        print("Evaluating requests")
        for row in load_csv_rows("./test/qa/QA_billability.csv"):
            query = row['question']
            
            try:
                response = evaluator.evaluate_customer_request(
                    request=query,
                )
            except Exception as e:
                response = RequestEvaluationResult(
                    conclusion = "?",
                    reason = f"error execution response: {str(e).replace('\n', ' ')}"
                )

            results.append(dict(
                question=query,
                classification=billability_results.get(response.conclusion),
                reason=response.reason
            ))
        
        if settings.DEBUG:
            import json
            print(json.dumps(results, indent=4))

        print("Writing output to csv")
        write_dict_to_csv(
            path = "output.csv",
            data = results,
        )