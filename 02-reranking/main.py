"""
Qdrant Search Labs
Experiment 02

Hybrid Search + Cross-Encoder Reranking
"""

import os

from dotenv import load_dotenv
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient, models
from evaluation import (
    EVALUATION_QUERIES,
    recall_at_k,
    reciprocal_rank,
)


load_dotenv()


###############################################################
# Configuration
###############################################################

COLLECTION_NAME = "dense-sparse-hybrid"

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL = "Qdrant/bm25"

RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

#QUERY = "virtualization software container"

CANDIDATE_LIMIT = 6
FINAL_LIMIT = 3


###############################################################
# Connect
###############################################################

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)


###############################################################
# Reranker
###############################################################

reranker = TextCrossEncoder(
    model_name=RERANKER_MODEL
)

###############################################################
# Search Functions
###############################################################


def hybrid_search(query: str) -> list[models.ScoredPoint]:
    """Retrieve candidates using Dense + Sparse Search with RRF."""

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=models.Document(
                    text=query,
                    model=DENSE_MODEL,
                ),
                using="dense",
                limit=CANDIDATE_LIMIT,
            ),
            models.Prefetch(
                query=models.Document(
                    text=query,
                    model=SPARSE_MODEL,
                ),
                using="sparse",
                limit=CANDIDATE_LIMIT,
            ),
        ],
        query=models.RrfQuery(
            rrf=models.Rrf(
                k=2,
                weights=[1.0, 1.0],
            )
        ),
        limit=CANDIDATE_LIMIT,
    )

    return response.points


###############################################################
# Reranking Functions
###############################################################


def rerank(
    query: str,
    candidates: list[models.ScoredPoint],
) -> list[tuple[models.ScoredPoint, float]]:
    """Rerank RRF candidates using a Cross-Encoder."""

    documents = [
        candidate.payload["text"]
        for candidate in candidates
    ]

    scores = list(
        reranker.rerank(
            query,
            documents,
        )
    )

    ranking = sorted(
        range(len(candidates)),
        key=lambda i: scores[i],
        reverse=True,
    )

    return [
        (candidates[i], float(scores[i]))
        for i in ranking
    ]

###############################################################
# Main
###############################################################

rrf_rr_scores = []
rrf_recall_scores = []

cross_encoder_rr_scores = []
cross_encoder_recall_scores = []


for evaluation in EVALUATION_QUERIES:
    query = evaluation["query"]
    relevant_documents = evaluation["relevant"]

    print("\n" + "=" * 70)
    print(f"Query: {query}")
    print("=" * 70)

    # ---------------------------------------------------------
    # Hybrid Search (RRF)
    # ---------------------------------------------------------

    hybrid_results = hybrid_search(query)

    rrf_documents = [
        result.payload["text"]
        for result in hybrid_results
    ]

    rrf_rr = reciprocal_rank(
        rrf_documents,
        relevant_documents,
        k=FINAL_LIMIT,
    )

    rrf_recall = recall_at_k(
        rrf_documents,
        relevant_documents,
        k=FINAL_LIMIT,
    )

    rrf_rr_scores.append(rrf_rr)
    rrf_recall_scores.append(rrf_recall)

    # ---------------------------------------------------------
    # Cross-Encoder Reranking
    # ---------------------------------------------------------

    reranked_results = rerank(
        query,
        hybrid_results,
    )

    cross_encoder_documents = [
        result.payload["text"]
        for result, score in reranked_results
    ]

    cross_encoder_rr = reciprocal_rank(
        cross_encoder_documents,
        relevant_documents,
        k=FINAL_LIMIT,
    )

    cross_encoder_recall = recall_at_k(
        cross_encoder_documents,
        relevant_documents,
        k=FINAL_LIMIT,
    )

    cross_encoder_rr_scores.append(cross_encoder_rr)
    cross_encoder_recall_scores.append(cross_encoder_recall)

    # ---------------------------------------------------------
    # Results for this query
    # ---------------------------------------------------------

    print("\nRRF Top 3\n")

    for rank, result in enumerate(
        hybrid_results[:FINAL_LIMIT],
        start=1,
    ):
        print(f"{rank}. {result.payload['text']}")

    print(f"\nRR@3:     {rrf_rr:.4f}")
    print(f"Recall@3: {rrf_recall:.4f}")

    print("\nCross-Encoder Top 3\n")

    for rank, (result, score) in enumerate(
        reranked_results[:FINAL_LIMIT],
        start=1,
    ):
        print(f"{rank}. {result.payload['text']}")
        print(f"   Score: {score:.4f}")

    print(f"\nRR@3:     {cross_encoder_rr:.4f}")
    print(f"Recall@3: {cross_encoder_recall:.4f}")


###############################################################
# Final Evaluation
###############################################################

rrf_mrr = sum(rrf_rr_scores) / len(rrf_rr_scores)
rrf_mean_recall = sum(rrf_recall_scores) / len(rrf_recall_scores)

cross_encoder_mrr = (
    sum(cross_encoder_rr_scores)
    / len(cross_encoder_rr_scores)
)

cross_encoder_mean_recall = (
    sum(cross_encoder_recall_scores)
    / len(cross_encoder_recall_scores)
)


print("\n" + "=" * 70)
print("FINAL EVALUATION")
print("=" * 70)

print("\nRRF")
print(f"MRR@3:         {rrf_mrr:.4f}")
print(f"Mean Recall@3: {rrf_mean_recall:.4f}")

print("\nCross-Encoder")
print(f"MRR@3:         {cross_encoder_mrr:.4f}")
print(
    f"Mean Recall@3: "
    f"{cross_encoder_mean_recall:.4f}"
)
