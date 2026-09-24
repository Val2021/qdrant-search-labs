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
QUERY =  "DKR-6638"
#QUERY = "virtualization software"
#QUERY = "Using Podman as a Docker alternative"
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



def hybrid_search(query: str) -> list[models.ScoredPoint]:
    """Perform Hybrid Search using Reciprocal Rank Fusion (RRF)."""

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=models.Document(
                    text=query,
                    model=DENSE_MODEL,
                ),
                using="dense",
                limit=3,
            ),
            models.Prefetch(
                query=models.Document(
                    text=query,
                    model=SPARSE_MODEL,
                ),
                using="sparse",
                limit=3,
            ),
        ],
        query=models.FusionQuery(
            fusion=models.Fusion.RRF
        ),
        limit=3,
    )

    return response.points
###############################################################
# Main
###############################################################

print("=" * 60)
print(f"Query: {QUERY}")
print("=" * 60)

dense_results = dense_search(QUERY)

print("\nDense Search\n")

for rank, result in enumerate(dense_results, start=1):
    print(f"{rank}. {result.payload['text']}")
    print(f"   Dense Score: {result.score:.4f}")

sparse_results = sparse_search(QUERY)

print("\nSparse Search\n")

if not sparse_results:
    print("No matching documents found.")
else:
    for rank, result in enumerate(sparse_results, start=1):
        print(f"{rank}. {result.payload['text']}")
        print(f"   Sparse Score: {result.score:.4f}")

hybrid_results = hybrid_search(QUERY)

print("\nHybrid Search (RRF)\n")

for rank, result in enumerate(hybrid_results, start=1):
    print(f"{rank}. {result.payload['text']}")
    print(f"   RRF Score: {result.score:.4f}")
