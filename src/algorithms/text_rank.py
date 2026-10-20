"""
TextRank Graph Algorithm for Extractive Summarization
Builds a sentence graph with similarity edge weights and computes PageRank centrality.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TextRankSummarizer:
    """
    Extractive summarization using the PageRank graph algorithm on sentence embeddings.
    Formula:
    PR(u) = (1 - d) + d * sum_{v in In(u)} (w_{vu} / sum_{k} w_{vk}) * PR(v)
    where d = 0.85 (damping factor)
    """

    def __init__(self, damping: float = 0.85, max_iter: int = 50, tol: float = 1e-4):
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol

    def summarize(self, sentences: list[str], top_n: int = 4) -> list[tuple[int, float, str]]:
        """
        Extract the top_n most central/impactful sentences from the text.
        Returns:
            list of (original_index, textrank_score, sentence_text)
        """
        clean_sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 4]
        n = len(clean_sentences)

        if n <= top_n:
            return [(i, 1.0, s) for i, s in enumerate(clean_sentences)]

        # 1. Build TF-IDF sentence representations
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(clean_sentences)

        # 2. Compute similarity matrix (Adjacency Matrix W)
        sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Zero out diagonal to avoid self-loops
        np.fill_diagonal(sim_matrix, 0.0)

        # 3. Normalize row sums to get transition probabilities
        row_sums = sim_matrix.sum(axis=1, keepdims=True)
        # Handle disconnected nodes
        row_sums[row_sums == 0] = 1.0
        transition_matrix = sim_matrix / row_sums

        # 4. Power iteration for PageRank
        scores = np.ones(n) / n
        for _ in range(self.max_iter):
            next_scores = (1 - self.damping) / n + self.damping * transition_matrix.T.dot(scores)
            if np.linalg.norm(next_scores - scores, 1) < self.tol:
                break
            scores = next_scores

        # Rank sentences by score
        ranked_indices = np.argsort(scores)[::-1][:top_n]
        
        # Sort top sentences by their original chronological flow for natural reading
        chronological_top = sorted(ranked_indices)

        return [
            (int(idx), float(scores[idx]), clean_sentences[idx])
            for idx in chronological_top
        ]
