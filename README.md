# Hackathon onPrem-GPT

## Usage
1. Load the documents
```bash
python main.py init
```

2. Ask questions: Just run the command
It'll load the `questions` CSV column (important - encoding to read is configured based on example file; might need to adjust encoding in fileio/csv.py)
```bash
python main.py
```
The result will automatically be written into output.csv

Note: To receive debug output, set DEBUG=true
```
export DEBUG=true
```

# Pre-requisites

- Python 3.11 (https://www.python.org/downloads/release/python-31110/)
- Ollama (https://ollama.com/)
- nlm-ingestor running for document parsing (https://github.com/nlmatics/nlm-ingestor/)
> Pull the docker image
> ```
> docker pull ghcr.io/nlmatics/nlm-ingestor:latest
> ```
> Run the docker image mapping the port 5001 to port of your choice. 
> ```
> docker run -p 5010:5001 ghcr.io/nlmatics/nlm-ingestor:latest-<version>
> ```

## Installation

optional: create and activate an environment with the following command:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

install the requirements:
```bash
pip install -r requirements.txt
```

## Ollama

To install ollama go to:
https://ollama.com/

if you have to pull the ollama embedding model jina/jina-embeddings-v2-base-en:latest then run
    
```bash
ollama pull jina/jina-embeddings-v2-base-en:latest
```

for llama3.2:3b run the following command:
```bash
  ollama pull llama3.2:3b 
```