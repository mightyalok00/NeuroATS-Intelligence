"""Reproducible ranking evaluation for NeuroATS retrieval methods.

Compares keyword overlap, BM25, and TF-IDF cosine similarity on a larger,
fixed synthetic corpus. Relevance labels are provisional author-created labels;
this benchmark is not real-world resume/job validation.

Run: python -m benchmarks.evaluate_ranking
Report: python -m benchmarks.evaluate_ranking --output reports/ranking_evaluation.md
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.feature_engineering.vectorizers import BM25Okapi

# Synthetic role/profile snippets spanning software, data, cloud, security,
# testing, and ML. These are controlled fixtures, not real candidate resumes.
DOCUMENTS = [
    {"id": "D01", "text": "Python FastAPI backend REST APIs PostgreSQL Redis microservices"},
    {"id": "D02", "text": "Python Django backend web services SQL database migrations"},
    {"id": "D03", "text": "Java Spring Boot backend services Oracle enterprise APIs"},
    {"id": "D04", "text": "React TypeScript frontend components CSS accessibility web application"},
    {"id": "D05", "text": "Vue JavaScript frontend UI design responsive browser applications"},
    {"id": "D06", "text": "Angular TypeScript frontend forms state management web interfaces"},
    {"id": "D07", "text": "PyTorch machine learning NLP transformers model training"},
    {"id": "D08", "text": "TensorFlow deep learning neural networks image classification"},
    {"id": "D09", "text": "Computer vision image segmentation OpenCV object detection"},
    {"id": "D10", "text": "Pandas NumPy Python data analysis exploratory statistics notebooks"},
    {"id": "D11", "text": "SQL analytics dashboards business intelligence reporting metrics"},
    {"id": "D12", "text": "Spark distributed data engineering ETL pipelines lakehouse"},
    {"id": "D13", "text": "AWS Docker Kubernetes deployment infrastructure observability"},
    {"id": "D14", "text": "Terraform infrastructure as code cloud provisioning AWS"},
    {"id": "D15", "text": "Azure cloud administration identity networking virtual machines"},
    {"id": "D16", "text": "CI CD GitHub Actions automated deployment release pipelines"},
    {"id": "D17", "text": "Cybersecurity threat modeling incident response SIEM monitoring"},
    {"id": "D18", "text": "Application security penetration testing OWASP vulnerability assessment"},
    {"id": "D19", "text": "Quality assurance test automation Selenium browser regression testing"},
    {"id": "D20", "text": "Pytest unit testing API integration tests fixtures coverage"},
    {"id": "D21", "text": "Product management roadmap user research prioritization stakeholder alignment"},
    {"id": "D22", "text": "UX research usability testing wireframes prototypes interaction design"},
    {"id": "D23", "text": "Technical writing API documentation developer guides information architecture"},
    {"id": "D24", "text": "Recruiting talent sourcing interview coordination applicant tracking"},
    {"id": "D25", "text": "MLOps model serving experiment tracking model monitoring MLflow"},
    {"id": "D26", "text": "Natural language processing text classification named entity recognition"},
    {"id": "D27", "text": "Database administration PostgreSQL backups query optimization replication"},
    {"id": "D28", "text": "Network engineering TCP IP routing firewalls network troubleshooting"},
    {"id": "D29", "text": "Financial analysis forecasting Excel budgeting variance analysis"},
    {"id": "D30", "text": "Marketing SEO keyword research search analytics content optimization"},
    {"id": "D31", "text": "Mobile development Android Kotlin application lifecycle"},
    {"id": "D32", "text": "iOS development Swift UIKit mobile applications"},
    {"id": "D33", "text": "Data visualization Tableau Power BI dashboards reporting"},
    {"id": "D34", "text": "Natural language generation LLM evaluation prompt testing retrieval augmented generation"},
    {"id": "D35", "text": "Linux systems administration shell scripting server troubleshooting"},
    {"id": "D36", "text": "Scrum agile delivery sprint planning retrospectives team facilitation"},
]

# Fixed query set and binary relevance labels. Each query has two relevant
# examples where possible. These labels need independent review before any
# external claim; see docs/BENCHMARK_LABEL_REVIEW.md.
HELD_OUT_QUERIES = [
    {"id": "Q01", "query": "Python backend REST API SQL", "relevant": {"D01", "D02"}},
    {"id": "Q02", "query": "Java enterprise backend services", "relevant": {"D03", "D01"}},
    {"id": "Q03", "query": "TypeScript frontend web UI", "relevant": {"D04", "D06"}},
    {"id": "Q04", "query": "responsive JavaScript frontend", "relevant": {"D05", "D04"}},
    {"id": "Q05", "query": "PyTorch NLP machine learning", "relevant": {"D07", "D26"}},
    {"id": "Q06", "query": "image classification computer vision", "relevant": {"D08", "D09"}},
    {"id": "Q07", "query": "Python pandas data analysis", "relevant": {"D10", "D11"}},
    {"id": "Q08", "query": "SQL business intelligence dashboards", "relevant": {"D11", "D33"}},
    {"id": "Q09", "query": "Spark ETL data pipelines", "relevant": {"D12", "D10"}},
    {"id": "Q10", "query": "AWS Docker Kubernetes cloud deployment", "relevant": {"D13", "D14"}},
    {"id": "Q11", "query": "Terraform cloud infrastructure provisioning", "relevant": {"D14", "D13"}},
    {"id": "Q12", "query": "Azure networking cloud administration", "relevant": {"D15", "D28"}},
    {"id": "Q13", "query": "GitHub Actions CI CD deployment", "relevant": {"D16", "D13"}},
    {"id": "Q14", "query": "security incident response SIEM", "relevant": {"D17", "D18"}},
    {"id": "Q15", "query": "OWASP application penetration testing", "relevant": {"D18", "D17"}},
    {"id": "Q16", "query": "Selenium automated browser regression tests", "relevant": {"D19", "D20"}},
    {"id": "Q17", "query": "pytest API unit integration testing", "relevant": {"D20", "D19"}},
    {"id": "Q18", "query": "product roadmap user research prioritization", "relevant": {"D21", "D22"}},
    {"id": "Q19", "query": "UX usability wireframes prototypes", "relevant": {"D22", "D21"}},
    {"id": "Q20", "query": "developer guides API documentation", "relevant": {"D23", "D01"}},
    {"id": "Q21", "query": "MLOps model serving monitoring MLflow", "relevant": {"D25", "D07"}},
    {"id": "Q22", "query": "LLM retrieval augmented generation evaluation", "relevant": {"D34", "D26"}},
    {"id": "Q23", "query": "Linux shell scripting server administration", "relevant": {"D35", "D28"}},
    {"id": "Q24", "query": "SEO keyword research search analytics", "relevant": {"D30", "D33"}},
]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z0-9_+#.\-]+\b", text.lower())


def keyword_overlap_score(query: str, document: str) -> float:
    """Simple baseline: count unique query tokens present in the document."""
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
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2), stop_words="english", sublinear_tf=True
    )
    # Vectorizer is fitted only on the fixed document collection, not labels.
    document_matrix = tfidf.fit_transform(texts)
    methods = ("Keyword overlap", "BM25", "TF-IDF cosine")
    details: dict[str, list[dict]] = {method: [] for method in methods}

    for case in HELD_OUT_QUERIES:
        query, relevant = case["query"], case["relevant"]
        score_sets = {
            "Keyword overlap": np.array([keyword_overlap_score(query, text) for text in texts]),
            "BM25": bm25.get_scores(query),
            "TF-IDF cosine": (document_matrix @ tfidf.transform([query]).T).toarray().ravel(),
        }
        for method, scores in score_sets.items():
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
                "query_id": case["id"], "query": query,
                "relevant": sorted(relevant), "ranked": ranked,
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
        "# Ranking Evaluation Report", "",
        "> **Scope warning:** This benchmark has 36 synthetic snippets and 24 fixed queries. Labels are provisional author-created labels and have not yet been independently adjudicated. Results test retrieval metrics on a controlled fixture; they do not establish real-world resume/job accuracy, fairness, or hiring outcomes.",
        "", "## Protocol", "",
        "- Collection: 36 synthetic role/profile snippets across multiple technical and business domains.",
        "- Evaluation: 24 fixed queries with binary relevance labels; labels are versioned in this script.",
        "- Methods: unique-token overlap baseline, project BM25 implementation, and TF-IDF cosine similarity.",
        "- TF-IDF is fitted only on the document collection. Relevance labels are not model features.",
        "- Ties are broken deterministically by original document order.",
        "- Metrics: macro-averaged MRR, Recall@3, and NDCG@3.",
        "- Reproduce with: python -m benchmarks.evaluate_ranking",
        "- Label review process and status: docs/BENCHMARK_LABEL_REVIEW.md",
        "", "## Aggregate results", "",
        "| Method | MRR | Recall@3 | NDCG@3 |",
        "|---|---:|---:|---:|",
    ]
    for method, metrics in means.items():
        lines.append(f"| {method} | {metrics['mrr']:.3f} | {metrics['recall_at_3']:.3f} | {metrics['ndcg_at_3']:.3f} |")
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
        "- This larger fixture reduces dependence on a handful of examples but remains synthetic and small; metrics must not be generalized to real resumes or hiring decisions.",
        "- Inspect missed relevant documents query by query. Lexical methods can miss synonyms and may reward overlap without understanding role context.",
        "- Independent relevance review is still required. Do not describe labels as independently validated until reviewers complete the protocol and disagreements are recorded.",
        "- A real-world evaluation needs permitted representative resume/job data, candidate-level leakage controls, a frozen held-out test set, a documented rubric, independent annotators, uncertainty estimates, and job-family error analysis.",
        "- Do not use these scores to claim production quality, fairness, candidate quality, or hiring prediction.", "",
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
