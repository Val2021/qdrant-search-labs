"""
Qdrant Search Labs
Experiment 01

Dataset used to compare:

- Dense Search
- Sparse Search
- Hybrid Search (RRF)

The documents were intentionally chosen so that some queries
benefit from semantic search while others benefit from keyword search.
"""

documents = [
    "How to install Docker on Ubuntu",

    "Installing Docker Desktop on Windows",

    "Docker installation guide",

    "Using Podman as a Docker alternative",

    "Running Linux containers",

    "Container orchestration with Kubernetes",

    "Kubernetes deployment tutorial",

    "Building Docker images",

    "Deploying AI applications with Qdrant",

    "Vector databases for semantic search",
]
