import json
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct


BASE_DIR = Path(__file__).parent

DATA_FILE = BASE_DIR / "data" / "documents.jsonl"
EMBEDDINGS_FILE = BASE_DIR / "data" / "embeddings.npy"

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "documents"

BATCH_SIZE = 500


def load_documents() -> list[dict]:
    documents = []

    with DATA_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            documents.append(json.loads(line))

    return documents


def load_embeddings() -> np.ndarray:
    return np.load(EMBEDDINGS_FILE)


def main() -> None:
    documents = load_documents()
    embeddings = load_embeddings()

    print(f"Documents loaded: {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape}")

    if len(documents) != len(embeddings):
        raise ValueError(
            "Number of documents and embeddings does not match."
        )

    client = QdrantClient(url=QDRANT_URL)

    total = len(documents)

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)

        points = []

        for document, embedding in zip(
            documents[start:end],
            embeddings[start:end],
        ):
            points.append(
                PointStruct(
                    id=document["id"],
                    vector=embedding.tolist(),
                    payload={
                        "text": document["text"],
                        "tenant_id": document["tenant_id"],
                        "category": document["category"],
                        "language": document["language"],
                        "year": document["year"],
                    },
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        print(f"Uploaded {end}/{total} points")

    print("Qdrant load completed successfully.")


if __name__ == "__main__":
    main()
