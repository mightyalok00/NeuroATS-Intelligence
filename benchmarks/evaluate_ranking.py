"""Reproducible held-out ranking evaluation for NeuroATS retrieval methods.

Compares a unique-token-overlap baseline, the project's BM25 implementation,
and TF-IDF cosine similarity. The hand-labelled fixture validates the evaluation
pipeline, not real-world hiring performance.

Run from the repository root: python -m benchmarks.evaluate_ranking
Optional report file: python -m benchmarks.evaluate_ranking --output reports/ranking_evaluation.md
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.feature_engineering.vectorizers import BM25Okapi

DOCUMENTS = [
    {"id": "D1", "text": "Python FastAPI backend APIs PostgreSQL Redis microservices"},
    {"id": "D2", "text": "PyTorch machine learning NLP transformers model training"},
    {"id": "D3", "text": "React TypeScript frontend components CSS web application"},
    {"id": "D4", "text": "AWS Docker Kubernetes CI CD deployment infrastructure"},
    {"id": "D5", "text": "Python backend services REST API SQL database performance"},
    {"id": "D6", "text": "Computer vision image classification convolutional neural networks"},
    {"id": "D7", "text": "Data analyst SQL pandas dashboards statistics reporting"},
    {"id": "D8", "text": "Java Spring Boot backend services Oracle enterprise APIs"},
]

# Fixed held-out query set; labels are manually assigned for this illustrative fixture.
HELD_OUT_QUERIES = [
    {"id": "Q1", "query": "Python backend API SQL", "relevant": {"D1", "D5"}},
    {"id": "Q2", "query": "machine learning PyTorch NLP", "relevant": {"D2", "D6"}},
    {"id": "Q3", "query": "AWS Docker Kubernetes deployment", "relevant": {"D4"}},
    {"id": "Q4", "query": "SQL analytics dashboards data reporting", "relevant": {"D7"}},
]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z0-9_+#.\-]+\b", text.lower())


def keyword_overlap_score(query: str, document: str) -> float:
    """Simple baseline: number of unique query tokens present in the document."""
    return float(len(set(tokenize(query)) & set(tokenize(document))))


def rank_indices(scores: np.ndarray) -> list[int]:
    """Stable descending order; document order breaks ties deterministically."""
    return sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))


def reciprocal_rank(ranked_ids: list[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def ndcg_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked_ids[:k], start=1)
        if doc_id in relevant
    )
    ideal_count = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / idcg if idcg else 0.0


def evaluate() -> tuple[dict[str, list[dict]], dict[str, dict[str, float]]]:
    texts = [item["text"] for item in DOCUMENTS]
    ids = [item["id"] for item in DOCUMENTS]
    bm25 = BM25Okapi(texts)

    # Fit TF-IDF only on the retrieval collection, not on relevance labels.
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2), stop_words="english", sublinear_tf=True
    )
    document_matrix = tfidf.fit_transform(texts)

    methods = ("Keyword overlap", "BM25", "TF-IDF cosine")
    details: dict[str, list[dict]] = {method: [] for method in methods}

    for case in HELD_OUT_QUERIES:
        query = case["query"]
        relevant = case["relevant"]
        keyword_scores = np.array([keyword_overlap_score(query, text) for text in texts])
        bm25_scores = bm25.get_scores(query)
        query_vector = tfidf.transform([query])
        tfidf_scores = (document_matrix @ query_vector.T).toarray().ravel()

        all_scores = {
            "Keyword overlap": keyword_scores,
            "BM25": bm25_scores,
            "TF-IDF cosine": tfidf_scores,
        }
        for method, scores in all_scores.items():
            ranked = [ids[i] for i in rank_indices(scores)]
            metrics = {
                "mrr": reciprocal_rank(ranked, relevant),
                "recall_at_3": recall_at_k(ranked, relevant, 3),
                "ndcg_at_3": ndcg_at_k(ranked, relevant, 3),
            }
            for metric, value in metrics.items():
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise AssertionError(f"Invalid {metric}: {value}")
            details[method].append({
                "query_id": case["id"],
                "query": query,
                "relevant": sorted(relevant),
                "ranked": ranked,
                "top3": ranked[:3],
                "missed_relevant_at3": sorted(relevant - set(ranked[:3])),
                **metrics,
            })

    means = {
        method: {
            metric: sum(row[metric] for row in rows) / len(rows)
            for metric in ("mrr", "recall_at_3", "ndcg_at_3")
        }
        for method, rows in details.items()
    }
    return details, means


def render_report(details: dict[str, list[dict]], means: dict[str, dict[str, float]]) -> str:
    lines = [
        "# Ranking Evaluation Report",
        "",
        "> **Scope warning:** This is a reproducible toy benchmark with eight documents and four manually labelled held-out queries. It validates metric plumbing and compares retrieval methods on this fixture only. It is not evidence of resume/job matching accuracy, fairness, or hiring outcomes.",
        "",
        "## Protocol",
        "",
        "- Collection: eight short synthetic role/profile snippets.",
        "- Evaluation: four held-out queries with manually assigned binary relevance labels.",
        "- Methods: unique-token keyword overlap baseline, project BM25 implementation, and TF-IDF cosine similarity.",
        "- TF-IDF is fitted on the document collection only; held-out relevance labels are not used to fit retrieval models.",
        "- Ranking ties are broken deterministically by original document order.",
        "- Metrics: MRR, Recall@3, and NDCG@3; values are macro-averaged over queries.",
        "- Reproduce with: python -m benchmarks.evaluate_ranking",
        "",
        "## Aggregate results",
        "",
        "| Method | MRR | Recall@3 | NDCG@3 |",
        "|---|---:|---:|---:|",
    ]
    for method, metrics in means.items():
        lines.append(
            f"| {method} | {metrics['mrr']:.3f} | {metrics['recall_at_3']:.3f} | {metrics['ndcg_at_3']:.3f} |"
        )
    lines.extend(["", "## Per-query results and error analysis", ""])
    for method, rows in details.items():
        lines.extend([
            f"### {method}", "",
            "| Query | Relevant IDs | Top 3 | MRR | Recall@3 | NDCG@3 | Relevant IDs missed in top 3 |",
            "|---|---|---|---:|---:|---:|---|",
        ])
        for row in rows:
            lines.append(
                f"| {row['query_id']}: {row['query']} | {', '.join(row['relevant'])} | {', '.join(row['top3'])} | "
                f"{row['mrr']:.3f} | {row['recall_at_3']:.3f} | {row['ndcg_at_3']:.3f} | "
                f"{', '.join(row['missed_relevant_at3']) or 'None'} |"
            )
        lines.append("")
    lines.extend([
        "## Interpretation and limitations", "",
        "- Compare aggregate metrics only within this fixed fixture; four queries are far too few to support general conclusions.",
        "- Inspect missed relevant documents query by query. Lexical methods can miss synonyms and reward overlapping terms without understanding role context.",
        "- Labels are hand-authored and may encode author assumptions. Larger evaluations need independent annotators, a written rubric, adjudication, and agreement reporting.",
        "- A real evaluation should use permitted representative data; split by candidate and preferably source/time; freeze the test set before tuning; report uncertainty and job-family error analysis.",
        "- Do not use these scores to claim production quality, fairness, candidate quality, or hiring prediction.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional path to write the generated Markdown report.")
    args = parser.parse_args()

    details, means = evaluate()
    report = render_report(details, means)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
        print(f"\nWrote report to {args.output}")


if __name__ == "__main__":
    main()
