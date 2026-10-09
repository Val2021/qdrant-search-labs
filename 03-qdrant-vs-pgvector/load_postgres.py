import json
from pathlib import Path

import numpy as np
import psycopg


BASE_DIR = Path(__file__).parent

DATA_FILE = BASE_DIR / "data" / "documents.jsonl"
EMBEDDINGS_FILE = BASE_DIR / "data" / "embeddings.npy"


POSTGRES_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "vector_lab",
    "user": "postgres",
    "password": "postgres",
}


def load_documents() -> list[dict]:
    documents = []

    with DATA_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            documents.append(json.loads(line))

    return documents


def load_embeddings() -> np.ndarray:
    return np.load(EMBEDDINGS_FILE)


def clear_table(cursor) -> None:
    cursor.execute("TRUNCATE TABLE documents RESTART IDENTITY;")


def insert_documents(
    cursor,
    documents: list[dict],
    embeddings: np.ndarray,
) -> None:
    rows = []

    for document, embedding in zip(documents, embeddings):
        rows.append(
            (
                document["id"],
                document["text"],
                document["tenant_id"],
                document["category"],
                document["language"],
                document["year"],
                embedding.tolist(),
            )
        )

    cursor.executemany(
        """
        INSERT INTO documents (
            id,
            text,
            tenant_id,
            category,
            language,
            year,
            embedding
        )
        VALUES (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s
        );
        """,
        rows,
    )


def main() -> None:
    documents = load_documents()
    embeddings = load_embeddings()

    print(f"Documents loaded: {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape}")

    if len(documents) != len(embeddings):
        raise ValueError(
            "Number of documents and embeddings does not match."
        )

    with psycopg.connect(**POSTGRES_CONFIG) as connection:
        with connection.cursor() as cursor:
            clear_table(cursor)

            print("Inserting documents into PostgreSQL...")

            insert_documents(
                cursor,
                documents,
                embeddings,
            )

        connection.commit()

    print("PostgreSQL load completed successfully.")


if __name__ == "__main__":
    main()
