# 01 — Dense vs Sparse vs Hybrid Search

A small hands-on experiment comparing **Dense Search**, **Sparse Search**, and **Hybrid Search (RRF)** using Qdrant.

The dataset includes semantically related content and exact technical identifiers, allowing us to observe how different retrieval strategies behave with different types of queries.

## Stack

- **Vector Database:** Qdrant
- **Dense Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Sparse Model:** `Qdrant/bm25`
- **Fusion:** Reciprocal Rank Fusion (RRF)

## Project Structure

```text
01-dense-sparse-hybrid/
├── create_collection.py
├── dataset.py
├── upload_data.py
├── main.py
└── README.md
```

## Environment

Create a `.env` file in the project root:

```env
QDRANT_URL=https://YOUR-CLUSTER-URL
QDRANT_API_KEY=YOUR_API_KEY
```

See `.env.example` for reference.

## Running the Lab

Run the commands from the repository root.

### 1. Create the collection

```bash
uv run 01-dense-sparse-hybrid/create_collection.py
```

> This recreates the lab collection if it already exists.

### 2. Upload the dataset

```bash
uv run 01-dense-sparse-hybrid/upload_data.py
```

### 3. Run the experiment

```bash
uv run 01-dense-sparse-hybrid/main.py
```

Change `QUERY` in `main.py` to test different retrieval behaviors:

```python
QUERY = "docker installation"
# QUERY = "virtualization software"
# QUERY = "DKR-6638"
```

## What This Lab Demonstrates

- **Dense Search** — retrieves documents based on semantic similarity.
- **Sparse Search** — retrieves documents based on lexical relevance using BM25.
- **Hybrid Search** — combines both rankings using RRF.

**Key takeaway:** Dense captures meaning, Sparse preserves exact lexical signals, and Hybrid Search combines both when the dataset and queries require both types of retrieval.
