from __future__ import annotations

import time
from dataclasses import dataclass
from statistics import mean

import numpy as np
import psycopg
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    Range,
)


POSTGRES_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "vector_lab",
    "user": "postgres",
    "password": "postgres",
}

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "documents"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 10
WARMUP_RUNS = 5
BENCHMARK_RUNS = 100


@dataclass
class Scenario:
    name: str
    tenant_id: str | None = None
    category: str | None = None
    language: str | None = None
    min_year: int | None = None


SCENARIOS = [
    Scenario(
        name="no_filter",
    ),
    Scenario(
        name="category",
        category="python",
    ),
    Scenario(
        name="tenant",
        tenant_id="tenant_01",
    ),
    Scenario(
        name="tenant_category",
        tenant_id="tenant_01",
        category="python",
    ),
    Scenario(
        name="tenant_category_language_year",
        tenant_id="tenant_01",
        category="python",
        language="en",
        min_year=2025,
    ),
]


def generate_query_embedding(query: str) -> list[float]:
    model = TextEmbedding(model_name=MODEL_NAME)

    embedding = next(model.embed([query]))

    return np.asarray(
        embedding,
        dtype=np.float32,
    ).tolist()


def build_postgres_filter(
    scenario: Scenario,
) -> tuple[str, list]:
    conditions = []
    params = []

    if scenario.tenant_id is not None:
        conditions.append("tenant_id = %s")
        params.append(scenario.tenant_id)

    if scenario.category is not None:
        conditions.append("category = %s")
        params.append(scenario.category)

    if scenario.language is not None:
        conditions.append("language = %s")
        params.append(scenario.language)

    if scenario.min_year is not None:
        conditions.append("year >= %s")
        params.append(scenario.min_year)

    if not conditions:
        return "", params

    return "WHERE " + " AND ".join(conditions), params


def count_postgres_candidates(
    connection,
    scenario: Scenario,
) -> int:
    where_clause, params = build_postgres_filter(scenario)

    query = f"""
        SELECT COUNT(*)
        FROM documents
        {where_clause};
    """

    with connection.cursor() as cursor:
        cursor.execute(query, params)
        return cursor.fetchone()[0]


def search_postgres(
    connection,
    query_embedding: list[float],
    scenario: Scenario,
) -> list[tuple]:
    where_clause, filter_params = build_postgres_filter(
        scenario
    )

    query = f"""
        SELECT
            id,
            1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT {TOP_K};
    """

    params = [
        query_embedding,
        *filter_params,
        query_embedding,
    ]

    with connection.cursor() as cursor:
        cursor.execute(query, params)
        return cursor.fetchall()


def build_qdrant_filter(
    scenario: Scenario,
) -> Filter | None:
    must = []

    if scenario.tenant_id is not None:
        must.append(
            FieldCondition(
                key="tenant_id",
                match=MatchValue(
                    value=scenario.tenant_id
                ),
            )
        )

    if scenario.category is not None:
        must.append(
            FieldCondition(
                key="category",
                match=MatchValue(
                    value=scenario.category
                ),
            )
        )

    if scenario.language is not None:
        must.append(
            FieldCondition(
                key="language",
                match=MatchValue(
                    value=scenario.language
                ),
            )
        )

    if scenario.min_year is not None:
        must.append(
            FieldCondition(
                key="year",
                range=Range(
                    gte=scenario.min_year
                ),
            )
        )

    if not must:
        return None

    return Filter(must=must)


def search_qdrant(
    client: QdrantClient,
    query_embedding: list[float],
    scenario: Scenario,
):
    query_filter = build_qdrant_filter(scenario)

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        query_filter=query_filter,
        limit=TOP_K,
        with_payload=False,
    )

    return response.points


def measure_postgres(
    connection,
    query_embedding: list[float],
    scenario: Scenario,
) -> list[float]:
    times = []

    for _ in range(WARMUP_RUNS):
        search_postgres(
            connection,
            query_embedding,
            scenario,
        )

    for _ in range(BENCHMARK_RUNS):
        start = time.perf_counter()

        search_postgres(
            connection,
            query_embedding,
            scenario,
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        times.append(elapsed_ms)

    return times


def measure_qdrant(
    client: QdrantClient,
    query_embedding: list[float],
    scenario: Scenario,
) -> list[float]:
    times = []

    for _ in range(WARMUP_RUNS):
        search_qdrant(
            client,
            query_embedding,
            scenario,
        )

    for _ in range(BENCHMARK_RUNS):
        start = time.perf_counter()

        search_qdrant(
            client,
            query_embedding,
            scenario,
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        times.append(elapsed_ms)

    return times


def calculate_stats(
    times: list[float],
) -> dict:
    return {
        "mean": mean(times),
        "p50": np.percentile(times, 50),
        "p95": np.percentile(times, 95),
    }


def main() -> None:
    query = "How to build and deploy a Python API?"

    print(f"Query: {query}")
    print()

    query_embedding = generate_query_embedding(query)

    qdrant_client = QdrantClient(
        url=QDRANT_URL
    )

    with psycopg.connect(
        **POSTGRES_CONFIG
    ) as postgres_connection:

        total_documents = count_postgres_candidates(
            postgres_connection,
            Scenario(name="all"),
        )

        for scenario in SCENARIOS:
            print("=" * 80)
            print(f"Scenario: {scenario.name}")
            print("=" * 80)

            candidate_count = (
                count_postgres_candidates(
                    postgres_connection,
                    scenario,
                )
            )

            selectivity = (
                candidate_count
                / total_documents
                * 100
            )

            postgres_times = measure_postgres(
                postgres_connection,
                query_embedding,
                scenario,
            )

            qdrant_times = measure_qdrant(
                qdrant_client,
                query_embedding,
                scenario,
            )

            postgres_stats = calculate_stats(
                postgres_times
            )

            qdrant_stats = calculate_stats(
                qdrant_times
            )

            print(
                f"Candidates: {candidate_count}"
            )
            print(
                f"Selectivity: "
                f"{selectivity:.2f}%"
            )

            print()

            print("PostgreSQL / pgvector")
            print(
                f"  mean: "
                f"{postgres_stats['mean']:.2f} ms"
            )
            print(
                f"  p50:  "
                f"{postgres_stats['p50']:.2f} ms"
            )
            print(
                f"  p95:  "
                f"{postgres_stats['p95']:.2f} ms"
            )

            print()

            print("Qdrant")
            print(
                f"  mean: "
                f"{qdrant_stats['mean']:.2f} ms"
            )
            print(
                f"  p50:  "
                f"{qdrant_stats['p50']:.2f} ms"
            )
            print(
                f"  p95:  "
                f"{qdrant_stats['p95']:.2f} ms"
            )

            print()


if __name__ == "__main__":
    main()
