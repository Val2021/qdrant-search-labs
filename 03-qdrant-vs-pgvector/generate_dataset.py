import json
import random
from itertools import product
from pathlib import Path


TOTAL_DOCUMENTS = 10_000
RANDOM_SEED = 42

OUTPUT_FILE = Path(__file__).parent / "data" / "documents.jsonl"


TENANTS = [f"tenant_{i:02d}" for i in range(1, 21)]

CATEGORIES = [
    "python",
    "database",
    "cloud",
    "ai",
]

LANGUAGES = [
    "en",
    "pt",
    "de",
]

YEARS = [
    2023,
    2024,
    2025,
    2026,
]


TOPICS = {
    "python": [
        "FastAPI application development",
        "Python asynchronous programming",
        "REST API design with Python",
        "Python testing with pytest",
        "Python dependency management",
    ],
    "database": [
        "PostgreSQL query optimization",
        "database indexing strategies",
        "SQL performance tuning",
        "database replication",
        "transaction management",
    ],
    "cloud": [
        "Docker container deployment",
        "Kubernetes orchestration",
        "cloud infrastructure automation",
        "microservice deployment",
        "container monitoring",
    ],
    "ai": [
        "vector search",
        "retrieval augmented generation",
        "embedding generation",
        "semantic search",
        "large language model applications",
    ],
}


TEXT_TEMPLATES = {
    "en": [
        "A practical guide to {topic}.",
        "Best practices for {topic} in production environments.",
        "Common challenges when working with {topic}.",
        "An introduction to {topic} for software engineers.",
        "Performance considerations for {topic}.",
    ],
    "pt": [
        "Um guia prático sobre {topic}.",
        "Boas práticas para trabalhar com {topic} em produção.",
        "Desafios comuns ao utilizar {topic}.",
        "Uma introdução a {topic} para engenheiros de software.",
        "Considerações de desempenho para {topic}.",
    ],
    "de": [
        "Ein praktischer Leitfaden zu {topic}.",
        "Best Practices für {topic} in Produktionsumgebungen.",
        "Häufige Herausforderungen bei {topic}.",
        "Eine Einführung in {topic} für Softwareentwickler.",
        "Performance-Aspekte bei {topic}.",
    ],
}


def generate_dataset() -> list[dict]:
    random.seed(RANDOM_SEED)

    combinations = list(
        product(
            TENANTS,
            CATEGORIES,
            LANGUAGES,
            YEARS,
        )
    )

    documents = []

    for document_id in range(1, TOTAL_DOCUMENTS + 1):
        tenant_id, category, language, year = combinations[
            (document_id - 1) % len(combinations)
        ]

        topic = random.choice(TOPICS[category])
        template = random.choice(TEXT_TEMPLATES[language])

        text = template.format(topic=topic)

        documents.append(
            {
                "id": document_id,
                "text": text,
                "tenant_id": tenant_id,
                "category": category,
                "language": language,
                "year": year,
            }
        )

    random.shuffle(documents)

    return documents


def save_dataset(documents: list[dict]) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for document in documents:
            file.write(json.dumps(document, ensure_ascii=False) + "\n")


def print_statistics(documents: list[dict]) -> None:
    print(f"Total documents: {len(documents)}")
    print(f"Tenants: {len(TENANTS)}")
    print(f"Categories: {len(CATEGORIES)}")
    print(f"Languages: {len(LANGUAGES)}")
    print(f"Years: {len(YEARS)}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    dataset = generate_dataset()
    save_dataset(dataset)
    print_statistics(dataset)
