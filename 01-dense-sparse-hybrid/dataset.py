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

    "SSD 200 GB vs 500 GB: Which one should I choose?",

    "Installing Docker Desktop on Windows",

    "Docker installation guide",

    "Using Podman as a Docker alternative",

    "Running Linux containers",

    "Container orchestration with Kubernetes",

    "Kubernetes deployment tutorial",

    "Building Docker images",

    "Deploying AI applications with Qdrant",

    "Vector databases for semantic search",

    "Docker error DKR-1042 occurs when the container engine cannot start.",

    "Docker error DKR-2087 occurs when the container engine cannot start.",

    "Docker error DKR-3914 occurs when the container engine cannot start.",

    "Docker error DKR-5521 occurs when the container engine cannot start.",

    "Docker error DKR-7810 occurs when the container engine cannot start.",

    "Docker error DKR-8452 occurs when the container engine cannot start.",

    "Docker error DKR-9017 occurs when the container engine cannot start.",
    
    "Docker error DKR-6638 occurs when the container engine cannot start.",
]
