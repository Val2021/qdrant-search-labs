"""
Evaluation set for Experiment 02.

Each query contains documents manually labeled as relevant.
"""

EVALUATION_QUERIES = [
    {
        "query": "docker alternative",
        "relevant": [
            "Using Podman as a Docker alternative",
        ],
    },
    {
        "query": "container orchestration",
        "relevant": [
            "Container orchestration with Kubernetes",
            "Kubernetes deployment tutorial",
        ],
    },
    {
        "query": "docker installation",
        "relevant": [
            "How to install Docker on Ubuntu",
            "Installing Docker Desktop on Windows",
            "Docker installation guide",
        ],
    },
    {
        "query": "DKR-6638",
        "relevant": [
            "Docker error DKR-6638 occurs when the container engine cannot start.",
        ],
    },
]


def reciprocal_rank(
    ranked_documents: list[str],
    relevant_documents: list[str],
    k: int = 3,
) -> float:
    """Calculate Reciprocal Rank at K."""

    for rank, document in enumerate(
        ranked_documents[:k],
        start=1,
    ):
        if document in relevant_documents:
            return 1.0 / rank

    return 0.0

def recall_at_k(
    ranked_documents: list[str],
    relevant_documents: list[str],
    k: int = 3,
) -> float:
    """Calculate Recall at K."""

    retrieved = ranked_documents[:k]

    relevant_retrieved = sum(
        document in relevant_documents
        for document in retrieved
    )

    return relevant_retrieved / len(relevant_documents)
