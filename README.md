# Hackathon onPrem-GPT

## Usage
1. Load the documents
```bash
python main.py init
```

2. Ask questions
...


# Pre-requisites

- Python 3.11 (https://www.python.org/downloads/release/python-31110/)
- Ollama (https://ollama.com/)

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