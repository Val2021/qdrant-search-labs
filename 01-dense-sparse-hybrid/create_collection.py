"""
Qdrant Search Labs
Experiment 01

Create the collection used for the Dense vs Sparse vs Hybrid Search demo.
"""

import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()

###############################################################
# Configuration
###############################################################

COLLECTION_NAME = "dense-sparse-hybrid"

###############################################################
# Connect to Qdrant
###############################################################

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

###############################################################
# Check if the collection already exists
###############################################################

collections = client.get_collections().collections

if any(collection.name == COLLECTION_NAME for collection in collections):
    print(f"Collection '{COLLECTION_NAME}' already exists.")
    raise SystemExit()

###############################################################
# Create Collection
###############################################################

client.create_collection(
    collection_name=COLLECTION_NAME,

    vectors_config={
        "dense": models.VectorParams(
            size=384,
            distance=models.Distance.COSINE,
        ),
    },

    sparse_vectors_config={
        "sparse": models.SparseVectorParams(
            modifier=models.Modifier.IDF,
        )
    },
)

print(f"Collection '{COLLECTION_NAME}' created successfully!")
