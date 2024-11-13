import os
import json
import numpy as np
import pandas as pd
from typing import List, Union, Generator, Iterator

from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

class RAG:

    class Parameters(BaseModel):
        LLAMAINDEX_OLLAMA_BASE_URL: str
        LLAMAINDEX_MODEL_NAME: str
        LLAMAINDEX_EMBEDDING_MODEL_NAME: str

    def __init__(self):
        self.documents = None
        self.index = None

        self.parameters = self.Parameters(
            LLAMAINDEX_OLLAMA_BASE_URL=os.getenv("LLAMAINDEX_OLLAMA_BASE_URL", "http://localhost:11434"),
            LLAMAINDEX_MODEL_NAME=os.getenv("LLAMAINDEX_MODEL_NAME", "llama3.2:1b"),
            LLAMAINDEX_EMBEDDING_MODEL_NAME=os.getenv("LLAMAINDEX_EMBEDDING_MODEL_NAME", "jina/jina-embeddings-v2-base-en:latest"),
        )

    def on_startup(self, model_id):
        # Initialize the embedding model and LLM
        embed_model = OllamaEmbedding(
            model_name=self.parameters.LLAMAINDEX_EMBEDDING_MODEL_NAME,
            base_url=self.parameters.LLAMAINDEX_OLLAMA_BASE_URL,
        )
        llm = Ollama(
            model=model_id,
            base_url=self.parameters.LLAMAINDEX_OLLAMA_BASE_URL,
        )

        # Load documents
        self.documents = SimpleDirectoryReader("test/dataset").load_data()

        # Create the index by passing llm and embed_model directly
        self.index = VectorStoreIndex.from_documents(
            self.documents,
            llm=llm,
            embed_model=embed_model
        )

    def on_shutdown(self):
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict, top_k: int = 5
    ) -> Union[str, Generator, Iterator]:
        # When creating the query engine, pass llm and embed_model if needed
        query_engine = self.index.as_query_engine(streaming=True, top_k=top_k)
        response = query_engine.query(user_message)
        return response.response_gen

def calculate_similarity(responses: List[str]) -> float:
    """Calculate the average cosine similarity among generated responses."""
    if len(responses) < 2:
        print("Not enough responses to calculate similarity.")
        return float('nan')
    vectorizer = TfidfVectorizer().fit_transform(responses)
    vectors = vectorizer.toarray()
    similarity_matrix = cosine_similarity(vectors)
    upper_triangle_indices = np.triu_indices_from(similarity_matrix, k=1)
    return np.mean(similarity_matrix[upper_triangle_indices])

def run_query_sync(query: str, model_size: str, top_k: int = 5) -> str:
    rag = RAG()
    rag.on_startup(model_id=model_size)

    user_message = query
    model_id = model_size
    messages = [{"role": "user", "content": user_message}]
    body = {}
    print(f"Running query: {user_message} with model: {model_id}")
    response_gen = rag.pipe(user_message, model_id, messages, body, top_k=top_k)
    response_text = ''.join(response_gen)

    rag.on_shutdown()
    return response_text

def evaluate_response_bias(query: str, iterations: int, model_sizes: List[str], top_k: int = 5) -> dict:
    similarity_results = {}

    for model_size in model_sizes:
        all_responses = []

        for _ in range(iterations):
            print(f"Running query for model: {model_size}")
            response = run_query_sync(query, model_size, top_k)
            all_responses.append(response)

        if len(all_responses) < 2:
            print(f"Not enough responses for model {model_size} to calculate similarity.")
            avg_similarity = float('nan')
        else:
            print(f"Calculating average response similarity for model: {model_size}")
            avg_similarity = calculate_similarity(all_responses)

        similarity_results[model_size] = avg_similarity
        print(f"Model: {model_size}, Average Response Similarity: {avg_similarity:.3f}")

    return similarity_results

def main():
    # Parameters
    prompt = """
        Classify the following request among the following request categories:
            - Emergency Maintenance (IMBL Scanner Breakdown - Immediate Repair Required)
            - Routine Maintenance (Scheduled Maintenance Request for IMBL Scanner)
            - Upgrades (Request for Software Upgrade on IMBL Scanner)
            - Training (Request for Additional IMBL Scanner Training)
            - Replacement Under Warranty (Request for Replacement of Defective IMBL Scanner Part)
            - Customization (Request for Customization of IMBL Scanner Settings)
            - User-Caused Damage (Repair Request for IMBL Scanner - Accidental Damage)
            - New Projects (Request for Installation and Setup of New IMBL Scanner; Request for Remote Troubleshooting - IMBL Scanner Software Issue)
            - Performance Reports (Request for Detailed Monthly Performance Report)
            - Other (all other requests)

        Then check if the following request is covered under the Service Level Agreement (SLA) or not. Return the relevant passages from the SLA under "context" below.

        Then decide if the request is covered by the SLA, in that case the request is Billable, else the request is Not Billable. Also consider the Billable / Not Billable information depending on the request category.

        Return your reply in the following JSON format, nothing else, and follow strictly this format:
        {
            category: [one of the request category above],
            context: [relevant passage from the Service Level Agreement, do not include the customer request here],
            billable: [Billable / Not Billable]
        }
    """

    max_iterations = 80  # Define maximum number of iterations for running the test
    model_sizes = [
        "llama3.2:1b"
        "llama3.2:3b",
        "llama3.1:8b"
    ]

    # Colormap to create a gradient from red to blue
    cmap = plt.get_cmap("coolwarm", max_iterations)

    # Plotting
    plt.figure(figsize=(10, 6))

    # Run the similarity test for each iteration count
    for i in range(2, max_iterations + 1):  # Start from 2 to ensure at least two responses
        # Calculate similarity results for the current number of iterations
        similarity_results = evaluate_response_bias(prompt, i, model_sizes)
        similarities = list(similarity_results.values())

        # Mask the nan values in similarities
        masked_similarities = np.ma.masked_invalid(similarities)

        # Plot with color corresponding to the iteration number
        plt.plot(
            model_sizes,
            masked_similarities,
            marker='o',
            linestyle='-',
            color=cmap(i - 2),  # Adjust index to match colormap range
            linewidth=2,
            markersize=6,
            label=f'Iterations: {i}'
        )

    # Labels and Title
    plt.xlabel("Model Size", fontsize=14)
    plt.ylabel("Average Response Similarity", fontsize=14)
    plt.title("Average Response Similarity Across Model Sizes and Iterations", fontsize=16)

    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha="right")

    # Show grid and legend
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(title="Iteration Count", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Display the plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
