"""
Qdrant Search Labs
Experiment 01

Dense Search vs Sparse Search
"""

import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()

###############################################################
# Configuration
###############################################################

COLLECTION_NAME = "dense-sparse-hybrid"

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL = "Qdrant/bm25"

#QUERY = "docker installation"
#QUERY = "container engine"
QUERY = "virtualization software"
###############################################################
# Connect
###############################################################

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

###############################################################
# Search Functions
###############################################################

def dense_search(query: str) -> list[models.ScoredPoint]:
    """Perform Dense Search."""

    response = client.query_points(
        collection_name="dense-sparse-hybrid",
        query=models.Document(
            text=query,
            model= "sentence-transformers/all-MiniLM-L6-v2",
        ),
        using="dense",
        limit=3,
    )

    return response.points


def sparse_search(query: str) -> list[models.ScoredPoint]:
    """Perform Sparse Search."""

    response = client.query_points(
        collection_name="dense-sparse-hybrid",
        query=models.Document(
            text=query,
            model="Qdrant/bm25",
        ),
        using="sparse",
        limit=3,
    )

    return response.points


###############################################################
# Main
###############################################################

print("=" * 60)
print(f"Query: {QUERY}")
print("=" * 60)

print("\nDense Search\n")

for result in dense_search(QUERY):
    print(f"- {result.payload['text']}")
    print(f"  Score: {result.score:.4f}")

print("\nSparse Search\n")

for result in sparse_search(QUERY):
    print(f"- {result.payload['text']}")
    print(f"  Score: {result.score:.4f}")
