import psycopg


POSTGRES_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "vector_lab",
    "user": "postgres",
    "password": "postgres",
}


def setup_indexes() -> None:
    with psycopg.connect(**POSTGRES_CONFIG) as connection:
        with connection.cursor() as cursor:
            print("Creating HNSW index...")

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_documents_embedding_hnsw
                ON documents
                USING hnsw (embedding vector_cosine_ops);
                """
            )

            print("Creating metadata indexes...")

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_documents_tenant_id
                ON documents (tenant_id);
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_documents_category
                ON documents (category);
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_documents_language
                ON documents (language);
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_documents_year
                ON documents (year);
                """
            )

        connection.commit()

    print("PostgreSQL indexes created successfully.")


if __name__ == "__main__":
    setup_indexes()
