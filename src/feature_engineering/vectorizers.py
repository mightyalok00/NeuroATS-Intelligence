"""
Vectorization and BM25 Okapi Ranking Module
Implements TF-IDF n-gram vectorizer and BM25 Okapi algorithm in pure NumPy.
"""

import math
import re
from typing import Dict, List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class BM25Okapi:
    """
    BM25 Okapi implementation for exact relevance ranking over document chunks/sentences.
    Formula:
    Score(D, Q) = sum_{q in Q} IDF(q) * (f(q, D) * (k1 + 1)) / (f(q, D) + k1 * (1 - b + b * (|D| / avgdl)))
    IDF(q) = ln((N - n(q) + 0.5) / (n(q) + 0.5) + 1)
    """

    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.corpus_size = len(corpus)
        self.tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        self.doc_lens = [len(doc) for doc in self.tokenized_corpus]
        self.avgdl = sum(self.doc_lens) / max(self.corpus_size, 1)
        self.idf: Dict[str, float] = {}
        self.doc_freqs: List[Dict[str, int]] = []
        self._initialize()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9_\-+#.]+\b", text.lower())

    def _initialize(self):
        nd: Dict[str, int] = {}
        for doc in self.tokenized_corpus:
            frequencies = {}
            for word in doc:
                frequencies[word] = frequencies.get(word, 0) + 1
            self.doc_freqs.append(frequencies)
            for word in frequencies.keys():
                nd[word] = nd.get(word, 0) + 1

        for word, freq in nd.items():
            # Robertson-Spärck Jones IDF formula
            idf_val = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)
            self.idf[word] = max(idf_val, 0.05)

    def get_scores(self, query: str) -> np.ndarray:
        """Calculate BM25 relevance score for each document in the corpus."""
        query_tokens = self._tokenize(query)
        scores = np.zeros(self.corpus_size)

        for q in query_tokens:
            q_idf = self.idf.get(q, 0.0)
            if q_idf == 0.0:
                continue

            for idx, doc_freq in enumerate(self.doc_freqs):
                freq = doc_freq.get(q, 0)
                if freq == 0:
                    continue
                doc_len = self.doc_lens[idx]
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * (doc_len / max(self.avgdl, 1)))
                scores[idx] += q_idf * (numerator / denominator)

        return scores

    def rank(self, query: str, top_k: int = 5) -> List[Tuple[int, float, str]]:
        """Return top_k ranked documents with indices and scores."""
        scores = self.get_scores(query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((int(idx), float(scores[idx]), self.corpus[idx]))
        return results


class TFIDFEngine:
    """Wrapper for TF-IDF feature extraction with unigram and bigram representation."""

    def __init__(self, max_features: int = 2000):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english",
            max_features=max_features
        )
        self.is_fitted = False

    def fit_transform(self, corpus: List[str]):
        self.is_fitted = True
        return self.vectorizer.fit_transform(corpus)

    def transform(self, texts: List[str]):
        if not self.is_fitted:
            raise ValueError("TFIDFEngine must be fitted before transforming.")
        return self.vectorizer.transform(texts)

    def get_top_terms(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Extract highest weighted TF-IDF terms from a single document."""
        if not self.is_fitted:
            self.fit_transform([text])
            
        vec = self.vectorizer.transform([text]).toarray()[0]
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        top_indices = np.argsort(vec)[::-1][:top_n]
        
        return [
            (str(feature_names[i]), float(vec[i]))
            for i in top_indices
            if vec[i] > 0
        ]
