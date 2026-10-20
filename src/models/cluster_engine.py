"""
K-Means Semantic Clustering Engine for Bullet Points and Experience
"""

from typing import Dict, List
import numpy as np
from sklearn.cluster import KMeans
from src.feature_engineering.embeddings import DenseEmbedder


class ExperienceClusterEngine:
    """Clusters resume bullet points into thematic operational categories using K-Means."""

    def __init__(self, embedder: DenseEmbedder):
        self.embedder = embedder

    def cluster_bullets(self, bullets: List[str], max_clusters: int = 4) -> List[Dict[str, any]]:
        """Cluster bullets into cohesive thematic work streams."""
        valid_bullets = [b.strip() for b in bullets if len(b.strip().split()) >= 4]
        if len(valid_bullets) < 2:
            return [{
                "cluster_id": 1,
                "label": "Core Experience",
                "bullets": valid_bullets,
                "size": len(valid_bullets)
            }]

        k = min(max_clusters, max(2, len(valid_bullets) // 2))
        embeddings = self.embedder.encode(valid_bullets)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        clusters: Dict[int, List[str]] = {i: [] for i in range(k)}
        for bullet, label in zip(valid_bullets, labels):
            clusters[label].append(bullet)

        results: List[Dict[str, any]] = []
        for c_id, c_bullets in clusters.items():
            if not c_bullets:
                continue
            # Pick a cluster label by finding central bullet or key verbs
            first_words = [b.split()[0].title() for b in c_bullets if b.split()]
            summary_label = f"Stream {c_id + 1} ({', '.join(first_words[:2])})" if first_words else f"Workstream {c_id + 1}"

            results.append({
                "cluster_id": c_id + 1,
                "label": summary_label,
                "bullets": c_bullets,
                "size": len(c_bullets)
            })

        return results
