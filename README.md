# Hackathon onPrem-GPT

# Pre-requisites

- Python 3.11 (https://www.python.org/downloads/release/python-31110/)
- Conda (https://www.anaconda.com/download)
- Ollama (https://ollama.com/)
- PyCharm (optional)

## Installation

create a conda environment with the following command:
```bash
conda create -n rag python=3.11.10
```
activate the environment:
```bash
conda activate rag
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
ollama run jina/jina-embeddings-v2-base-en:latest
```

for llama3.2:3b run the following command:
```bash
  ollama run llama3.2:3b 
```


Note: focus was on accuracy (rather than performance), to handle reproducibilty a version with 3 runs has been investigated (final version main.py)