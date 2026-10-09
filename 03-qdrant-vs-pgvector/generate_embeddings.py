import json
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding


BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data" / "documents.jsonl"
OUTPUT_FILE = BASE_DIR / "data" / "embeddings.npy"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents() -> list[dict]:
    documents = []

    with DATA_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            documents.append(json.loads(line))

    return documents


def generate_embeddings(documents: list[dict]) -> np.ndarray:
    model = TextEmbedding(model_name=MODEL_NAME)

    texts = [document["text"] for document in documents]

    embeddings = list(model.embed(texts))

    return np.array(embeddings, dtype=np.float32)


def save_embeddings(embeddings: np.ndarray) -> None:
    np.save(OUTPUT_FILE, embeddings)


def main() -> None:
    documents = load_documents()

    print(f"Documents loaded: {len(documents)}")
    print(f"Generating embeddings with: {MODEL_NAME}")

    embeddings = generate_embeddings(documents)

    print(f"Embeddings shape: {embeddings.shape}")

    save_embeddings(embeddings)

    print(f"Embeddings saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
