"""
Qdrant Search Labs
Experiment 01

Upload the demo dataset to Qdrant.
"""

import os
from uuid import uuid4

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from dataset import documents

load_dotenv()

###############################################################
# Configuration
###############################################################

COLLECTION_NAME = "dense-sparse-hybrid"

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL = "Qdrant/bm25"

###############################################################
# Connect to Qdrant
###############################################################

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

###############################################################
# Upload Documents
###############################################################

points = []

for document in documents:

    points.append(
        models.PointStruct(
            id=uuid4().hex,
            vector={
                "dense": models.Document(
                    text=document,
                    model=DENSE_MODEL,
                ),
                "sparse": models.Document(
                    text=document,
                    model=SPARSE_MODEL,
                ),
            },
            payload={
                "text": document,
            },
        )
    )

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points,
)

###############################################################
# Done
###############################################################

print(f"Uploaded {len(points)} documents successfully!")
