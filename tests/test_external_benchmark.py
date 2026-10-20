import csv
import tempfile
import unittest
from pathlib import Path

from benchmarks.evaluate_external_dataset import evaluate_rows, load_rows


def sample_rows():
    return [
        {"query_id": "q1", "query_text": "python backend", "document_id": "d1", "candidate_id": "c1", "document_text": "Python backend API development", "relevance": "1", "split": "test"},
        {"query_id": "q1", "query_text": "python backend", "document_id": "d2", "candidate_id": "c2", "document_text": "Frontend React design", "relevance": "0", "split": "test"},
        {"query_id": "q2", "query_text": "cloud infrastructure", "document_id": "d3", "candidate_id": "c3", "document_text": "AWS cloud infrastructure", "relevance": "1", "split": "test"},
        {"query_id": "q2", "query_text": "cloud infrastructure", "document_id": "d4", "candidate_id": "c4", "document_text": "Content writing and SEO", "relevance": "0", "split": "test"},
        {"query_id": "q3", "query_text": "Python", "document_id": "d5", "candidate_id": "c5", "document_text": "Python developer", "relevance": "1", "split": "train"},
    ]


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class TestExternalBenchmark(unittest.TestCase):
    def test_evaluation_returns_all_baselines_and_bounded_metrics(self):
        result = evaluate_rows(sample_rows(), k=2)
        self.assertEqual(result["test_queries_scored"], 2)
        self.assertEqual(set(result["aggregate"]), {"Keyword overlap", "BM25", "TF-IDF cosine"})
        for method_metrics in result["aggregate"].values():
            for value in method_metrics.values():
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)
        self.assertIn("do not establish", result["scope_warning"])

    def test_rejects_candidate_leakage_across_splits(self):
        rows = sample_rows()
        rows[0]["candidate_id"] = "same-candidate"
        rows[-1]["candidate_id"] = "same-candidate"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dataset.csv"
            write_rows(path, rows)
            with self.assertRaisesRegex(ValueError, "Candidate leakage"):
                load_rows(path)

    def test_rejects_duplicate_query_document_pairs(self):
        rows = sample_rows()
        rows.append(dict(rows[0]))
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dataset.csv"
            write_rows(path, rows)
            with self.assertRaisesRegex(ValueError, "duplicate query_id/document_id"):
                load_rows(path)

    def test_load_rows_requires_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dataset.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["query_id", "query_text"])
                writer.writeheader()
                writer.writerow({"query_id": "q1", "query_text": "Python"})
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                load_rows(path)


if __name__ == "__main__":
    unittest.main()
