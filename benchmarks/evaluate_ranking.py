"""Reproducible smoke benchmark for BM25 ranking.

This tiny, hand-labelled corpus validates ranking-metric plumbing only. It is
illustrative and is not evidence of real-world resume/job matching quality.
Run from the repository root with: python -m benchmarks.evaluate_ranking
"""

from __future__ import annotations

import math

from src.feature_engineering.vectorizers import BM25Okapi


CORPUS = [
    "Python FastAPI backend APIs PostgreSQL Redis microservices",
    "PyTorch machine learning NLP transformers model training",
    "React TypeScript frontend components CSS web application",
    "AWS Docker Kubernetes CI CD deployment infrastructure",
    "Python backend services REST API SQL database performance",
    "Computer vision image classification convolutional neural networks",
]

# Query labels are manually assigned for this toy fixture; relevant indices are
# known in advance so the script can verify metric calculations deterministically.
CASES = [
    {"name": "python backend", "query": "Python backend API SQL", "relevant": {0, 4}},
    {"name": "machine learning", "query": "machine learning PyTorch NLP", "relevant": {1, 5}},
    {"name": "cloud infrastructure", "query": "AWS Docker Kubernetes deployment", "relevant": {3}},
]


def reciprocal_rank(ranked: list[int], relevant: set[int]) -> float:
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def ndcg_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    dcg = sum(1.0 / math.log2(rank + 1)
              for rank, doc_id in enumerate(ranked[:k], start=1)
              if doc_id in relevant)
    ideal_relevant = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1)
               for rank in range(1, ideal_relevant + 1))
    return dcg / idcg if idcg else 0.0


def main() -> None:
    bm25 = BM25Okapi(CORPUS)
    results = []
    for case in CASES:
        ranked = [doc_id for doc_id, score, _ in bm25.rank(case["query"], top_k=len(CORPUS))]
        relevant = case["relevant"]
        results.append({
            "name": case["name"],
            "mrr": reciprocal_rank(ranked, relevant),
            "recall_at_3": recall_at_k(ranked, relevant, 3),
            "ndcg_at_3": ndcg_at_k(ranked, relevant, 3),
        })

    # Basic validity checks ensure the script fails loudly if metrics break.
    for row in results:
        for metric in ("mrr", "recall_at_3", "ndcg_at_3"):
            value = row[metric]
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise AssertionError(f"Invalid {metric} for {row['name']}: {value}")

    print("BM25 illustrative ranking smoke benchmark")
    print("WARNING: tiny hand-labelled fixture; not a real-world performance evaluation.")
    print(f"Queries: {len(results)} | Documents: {len(CORPUS)}")
    for row in results:
        print(
            f"- {row['name']}: MRR={row['mrr']:.3f}, "
            f"Recall@3={row['recall_at_3']:.3f}, NDCG@3={row['ndcg_at_3']:.3f}"
        )
    print("Mean metrics:")
    for metric in ("mrr", "recall_at_3", "ndcg_at_3"):
        mean = sum(row[metric] for row in results) / len(results)
        print(f"- {metric}: {mean:.3f}")


if __name__ == "__main__":
    main()
