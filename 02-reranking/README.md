# Experiment 02 — Cross-Encoder Reranking

This experiment extends the hybrid search pipeline from Experiment 01 by adding a Cross-Encoder reranking stage.

The goal is to observe what happens when candidates retrieved with Dense + Sparse Search and fused with Reciprocal Rank Fusion (RRF) are reranked using a model that evaluates the query and each candidate document together.

## Pipeline

```text
Query
  │
  ├── Dense Search ──┐
  │                  │
  └── Sparse Search ─┤
                     │
                     ▼
                    RRF
                     │
                     ▼
              Top 6 candidates
                     │
                     ▼
               Cross-Encoder
                     │
                     ▼
                  Top 3
```

The experiment reuses the Qdrant collection and dataset created in Experiment 01.

## Models

**Dense retrieval**

`sentence-transformers/all-MiniLM-L6-v2`

**Sparse retrieval**

`Qdrant/bm25`

**Cross-Encoder reranker**

`Xenova/ms-marco-MiniLM-L-6-v2`

The Cross-Encoder is executed locally through FastEmbed.

## Why Top 6 Candidates?

The dataset used in this lab contains only 19 documents.

Using a Top 20 candidate set would effectively mean reranking almost the entire corpus, which would not represent the intended retrieval → reranking pipeline.

For this small experiment, RRF therefore produces a shortlist of 6 candidates:

```python
CANDIDATE_LIMIT = 6
FINAL_LIMIT = 3
```

The Cross-Encoder reranks those same 6 candidates, and the final evaluation considers the Top 3.

## Cross-Encoder Reranking

Unlike the Dense retriever, which encodes queries and documents separately, the Cross-Encoder evaluates each query-document pair jointly.

Conceptually:

```text
(query, document 1) → relevance score
(query, document 2) → relevance score
(query, document 3) → relevance score
...
```

The candidates are then sorted by their Cross-Encoder scores.

The reranker does not retrieve new documents. It can only reorder the candidates already selected by the hybrid retrieval stage.

## Evaluation Set

To avoid judging ranking changes only by intuition, a small manually labeled evaluation set was created.

The evaluation contains four queries:

```text
docker alternative
container orchestration
docker installation
DKR-6638
```

For each query, one or more documents were manually labeled as relevant before calculating the evaluation metrics.

This is a small exploratory evaluation set, not a benchmark.

## Metrics

Two metrics are used.

### Reciprocal Rank at 3 — RR@3

RR@3 measures where the first relevant document appears within the Top 3.

```text
Relevant at position 1 → 1.0000
Relevant at position 2 → 0.5000
Relevant at position 3 → 0.3333
No relevant document   → 0.0000
```

The mean Reciprocal Rank across all evaluation queries gives MRR@3.

### Recall@3

Recall@3 measures how many of the documents labeled as relevant were retrieved within the Top 3.

For example, if two documents are relevant but only one appears in the Top 3:

```text
Recall@3 = 1 / 2 = 0.5
```

## Results

| Query | RRF RR@3 | RRF Recall@3 | Cross-Encoder RR@3 | Cross-Encoder Recall@3 |
|---|---:|---:|---:|---:|
| docker alternative | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| container orchestration | 1.0000 | 0.5000 | 1.0000 | 0.5000 |
| docker installation | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| DKR-6638 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Final results:

| Method | MRR@3 | Mean Recall@3 |
|---|---:|---:|
| RRF | 1.0000 | 0.8750 |
| RRF + Cross-Encoder | 1.0000 | 0.8750 |

## Ranking Changes

Although the final metrics were identical, the Cross-Encoder did change the ordering of candidates.

For example, for:

```text
docker installation
```

RRF produced:

```text
1. Docker installation guide
2. How to install Docker on Ubuntu
3. Installing Docker Desktop on Windows
```

After Cross-Encoder reranking:

```text
1. Docker installation guide
2. Installing Docker Desktop on Windows
3. How to install Docker on Ubuntu
```

All three documents were labeled as relevant, so the ranking change did not affect RR@3 or Recall@3.

This illustrates an important limitation of these metrics: they capture whether relevant documents appear and where the first relevant result occurs, but they do not distinguish different degrees of relevance between documents that share the same binary relevance label.

## Takeaway

In this small experiment, adding a Cross-Encoder changed the ranking but did not improve MRR@3 or Mean Recall@3.

This does not mean that Cross-Encoder reranking is ineffective. The result is specific to this small dataset, the selected queries, the manually assigned relevance labels, the candidate depth, and the chosen reranker model.

The experiment highlights an important point:

> A ranking change is not automatically a relevance improvement. Reranking should be evaluated against explicit relevance judgments rather than visual inspection alone.

A larger evaluation set and graded relevance metrics such as NDCG would be useful for a more complete comparison.

## Run

From the repository root:

```bash
uv run 02-reranking/main.py
```

The script runs the evaluation queries through both pipelines and reports per-query RR@3 and Recall@3 followed by the aggregate MRR@3 and Mean Recall@3.
