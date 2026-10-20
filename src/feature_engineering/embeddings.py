"""
Dense Semantic Embeddings and Cosine Similarity Engine
Wraps SentenceTransformer with an instant offline LSA/TF-IDF dense vector fallback.
"""

from typing import List, Optional
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


class DenseEmbedder:
    """Provides dense vector representations and cosine similarity calculations."""

    _instance = None
    _st_model = None

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_transformer: bool = True):
        self.model_name = model_name
        self.use_transformer = use_transformer
        self.fallback_vectorizer = None
        self.fallback_svd = None
        self._init_model()

    def _init_model(self):
        """Attempt to load sentence-transformers model; if fails or offline, use LSA fallback."""
        if self.use_transformer and DenseEmbedder._st_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                # Load with local cache or download once
                DenseEmbedder._st_model = SentenceTransformer(self.model_name)
            except Exception:
                DenseEmbedder._st_model = None

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of text strings into dense vector arrays."""
        if not texts:
            return np.zeros((0, 384))

        # Use transformer if loaded
        if DenseEmbedder._st_model is not None:
            try:
                embeddings = DenseEmbedder._st_model.encode(
                    texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True
                )
                return embeddings
            except Exception:
                pass

        # Robust, high-speed pure ML Fallback: LSA (TF-IDF + TruncatedSVD)
        return self._lsa_encode(texts)

    def _lsa_encode(self, texts: List[str]) -> np.ndarray:
        """Instant dense representation using Latent Semantic Analysis (LSA)."""
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=1000
        )
        tfidf_mat = vectorizer.fit_transform(texts)
        
        # Safeguard n_components for small matrices
        import warnings
        max_valid_components = min(tfidf_mat.shape[0], tfidf_mat.shape[1]) - 1
        n_components = max(1, min(128, max_valid_components)) if max_valid_components >= 1 else 1
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            dense_vecs = svd.fit_transform(tfidf_mat)
        
        # L2 Normalize
        norms = np.linalg.norm(dense_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return dense_vecs / norms

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute cosine similarity between two 1D vectors."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    @staticmethod
    def batch_cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        """Compute cosine similarities between query vector and matrix of doc vectors."""
        if len(doc_vecs) == 0:
            return np.array([])
        q_norm = np.linalg.norm(query_vec)
        if q_norm == 0:
            return np.zeros(len(doc_vecs))
        
        d_norms = np.linalg.norm(doc_vecs, axis=1)
        d_norms[d_norms == 0] = 1.0
        dots = np.dot(doc_vecs, query_vec)
        return dots / (d_norms * q_norm)
