"""Evaluate a permitted, externally supplied resume/job relevance CSV.

Expected columns: query_id, query_text, document_id, candidate_id, document_text,
relevance (0/1), split (train/dev/test). The script evaluates only split=test.
It does not download data, infer sensitive attributes, or claim validation.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.feature_engineering.vectorizers import BM25Okapi

REQUIRED_COLUMNS = {
    "query_id", "query_text", "document_id", "candidate_id",
    "document_text", "relevance", "split",
}


def load_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or []))
            raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV contains no data rows")

    candidate_splits: dict[str, set[str]] = defaultdict(set)
    query_texts: dict[str, set[str]] = defaultdict(set)
    doc_texts: dict[str, set[str]] = defaultdict(set)
    seen_pairs: set[tuple[str, str]] = set()
    for line_no, row in enumerate(rows, start=2):
        for field in REQUIRED_COLUMNS:
            if not (row.get(field) or "").strip():
                raise ValueError(f"Line {line_no}: {field} must not be empty")
        if row["split"] not in {"train", "dev", "test"}:
            raise ValueError(f"Line {line_no}: split must be train, dev, or test")
        if row["relevance"] not in {"0", "1"}:
            raise ValueError(f"Line {line_no}: relevance must be 0 or 1")
        pair = (row["query_id"], row["document_id"])
        if pair in seen_pairs:
            raise ValueError(f"Line {line_no}: duplicate query_id/document_id pair {pair}")
        seen_pairs.add(pair)
        candidate_splits[row["candidate_id"]].add(row["split"])
        query_texts[row["query_id"]].add(row["query_text"])
        doc_texts[row["document_id"]].add(row["document_text"])

    leaked = sorted(candidate for candidate, splits in candidate_splits.items() if len(splits) > 1)
    if leaked:
        raise ValueError(f"Candidate leakage across splits: {', '.join(leaked[:10])}")
    inconsistent_queries = sorted(q for q, texts in query_texts.items() if len(texts) > 1)
    if inconsistent_queries:
        raise ValueError(f"Query IDs have inconsistent text: {', '.join(inconsistent_queries[:10])}")
    inconsistent_docs = sorted(d for d, texts in doc_texts.items() if len(texts) > 1)
    if inconsistent_docs:
        raise ValueError(f"Document IDs have inconsistent text: {', '.join(inconsistent_docs[:10])}")
    if not any(row["split"] == "test" for row in rows):
        raise ValueError("No split=test rows found")
    return rows


def _metrics(ranked_ids: list[str], relevant: set[str], k: int) -> dict[str, float]:
    rr = next((1.0 / rank for rank, doc_id in enumerate(ranked_ids, 1) if doc_id in relevant), 0.0)
    recall = len(set(ranked_ids[:k]) & relevant) / len(relevant) if relevant else 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank, doc_id in enumerate(ranked_ids[:k], 1) if doc_id in relevant)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return {"mrr": rr, f"recall_at_{k}": recall, f"ndcg_at_{k}": dcg / ideal if ideal else 0.0}


def evaluate_rows(rows: list[dict[str, str]], k: int = 5) -> dict:
    if k < 1:
        raise ValueError("k must be at least 1")
    test_rows = [row for row in rows if row["split"] == "test"]
    queries: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in test_rows:
        queries[row["query_id"]].append(row)
    # The candidate documents in the held-out retrieval pool are indexed for
    # retrieval. No relevance labels are used to fit the vectorizer.
    docs_by_id: dict[str, str] = {}
    for row in test_rows:
        docs_by_id[row["document_id"]] = row["document_text"]
    doc_ids = sorted(docs_by_id)
    doc_texts = [docs_by_id[doc_id] for doc_id in doc_ids]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
    matrix = vectorizer.fit_transform(doc_texts)
    bm25 = BM25Okapi(doc_texts)
    methods: dict[str, list[dict[str, float]]] = {
        "Keyword overlap": [], "BM25": [], "TF-IDF cosine": []
    }
    per_query = []
    for query_id in sorted(queries):
        qrows = queries[query_id]
        query_text = qrows[0]["query_text"]
        relevant = {row["document_id"] for row in qrows if row["relevance"] == "1"}
        if not relevant:
            # Queries with no positive labels cannot meaningfully measure recall/NDCG.
            continue
        scores = {
            "Keyword overlap": np.array([
                len(set(query_text.lower().split()) & set(text.lower().split()))
                for text in doc_texts
            ], dtype=float),
            "BM25": bm25.get_scores(query_text),
            "TF-IDF cosine": (matrix @ vectorizer.transform([query_text]).T).toarray().ravel(),
        }
        query_result = {"query_id": query_id, "relevant_count": len(relevant), "methods": {}}
        for method, values in scores.items():
            ranked = [doc_ids[i] for i in sorted(range(len(values)), key=lambda i: (-float(values[i]), doc_ids[i]))]
            metrics = _metrics(ranked, relevant, k)
            methods[method].append(metrics)
            query_result["methods"][method] = {
                **metrics, f"top_{k}": ranked[:k],
                "missed_relevant": sorted(relevant - set(ranked[:k])),
            }
        per_query.append(query_result)
    if not per_query:
        raise ValueError("No test queries have at least one relevant document (relevance=1)")
    aggregate = {
        method: {
            metric: sum(row[metric] for row in metric_rows) / len(metric_rows)
            for metric in metric_rows[0]
        }
        for method, metric_rows in methods.items()
    }
    return {
        "scope_warning": "Results apply only to the supplied dataset and labels; they do not establish hiring accuracy, fairness, or job-performance validity.",
        "test_rows": len(test_rows),
        "test_queries_scored": len(per_query),
        "test_candidates": len({row["candidate_id"] for row in test_rows}),
        "k": k,
        "aggregate": aggregate,
        "per_query": per_query,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to an authorized CSV; never commit private data")
    parser.add_argument("--k", type=int, default=5, help="Ranking cutoff (default: 5)")
    parser.add_argument("--output", help="Optional Markdown report path")
    args = parser.parse_args()
    rows = load_rows(args.input)
    result = evaluate_rows(rows, k=args.k)
    lines = [
        "# External Dataset Ranking Evaluation", "",
        f"> **Scope warning:** {result['scope_warning']}", "",
        f"- Test rows: {result['test_rows']}",
        f"- Queries scored: {result['test_queries_scored']}",
        f"- Unique candidates in test retrieval pool: {result['test_candidates']}",
        f"- Cutoff: k={result['k']}", "",
        "| Method | MRR | Recall@k | NDCG@k |",
        "|---|---:|---:|---:|",
    ]
    for method, metrics in result["aggregate"].items():
        lines.append(f"| {method} | {metrics['mrr']:.3f} | {metrics[f'recall_at_{args.k}']:.3f} | {metrics[f'ndcg_at_{args.k}']:.3f} |")
    lines.extend(["", "## Per-query error analysis", ""])
    for query in result["per_query"]:
        lines.extend([f"### {query['query_id']}", ""])
        for method, metrics in query["methods"].items():
            lines.append(f"- **{method}:** top-{args.k}={metrics[f'top_{args.k}']}; missed relevant={metrics['missed_relevant']}")
        lines.append("")
    report = "\n".join(lines) + "\n"
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
