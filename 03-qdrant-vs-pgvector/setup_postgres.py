import psycopg


POSTGRES_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "vector_lab",
    "user": "postgres",
    "password": "postgres",
}


def setup_postgres() -> None:
    with psycopg.connect(**POSTGRES_CONFIG) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE EXTENSION IF NOT EXISTS vector;
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id BIGSERIAL PRIMARY KEY,
                    text TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    language TEXT NOT NULL,
                    year INTEGER NOT NULL,
                    embedding vector(384)
                );
                """
            )

        connection.commit()

    print("PostgreSQL setup completed successfully.")


if __name__ == "__main__":
    setup_postgres()
