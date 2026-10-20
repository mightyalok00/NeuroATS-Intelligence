"""
Unsupervised Topic Modeling Engine
Uses Latent Dirichlet Allocation (LDA) and NMF to discover latent skill themes.
"""

from typing import Any

from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class TopicModeler:
    """Discovers unsupervised thematic topic clusters from resume text and bullet points."""

    def __init__(self, n_topics: int = 3, method: str = "lda"):
        self.n_topics = n_topics
        self.method = method.lower()
        self.vectorizer = None
        self.model = None

    def fit_transform(self, documents: list[str]) -> list[dict[str, Any]]:
        """
        Fit topic model on documents/bullets and return discovered topics with top keywords.
        """
        valid_docs = [d.strip() for d in documents if len(d.strip().split()) >= 3]
        if len(valid_docs) < 2:
            return []

        actual_topics = min(self.n_topics, len(valid_docs))

        if self.method == "lda":
            self.vectorizer = CountVectorizer(
                ngram_range=(1, 2),
                stop_words="english",
                max_features=500,
                min_df=1
            )
            dtm = self.vectorizer.fit_transform(valid_docs)
            self.model = LatentDirichletAllocation(
                n_components=actual_topics,
                random_state=42,
                max_iter=20
            )
            doc_topic_dist = self.model.fit_transform(dtm)
        else:
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words="english",
                max_features=500,
                min_df=1
            )
            dtm = self.vectorizer.fit_transform(valid_docs)
            self.model = NMF(
                n_components=actual_topics,
                random_state=42,
                max_iter=100
            )
            doc_topic_dist = self.model.fit_transform(dtm)

        feature_names = self.vectorizer.get_feature_names_out()
        topics_info: list[dict[str, Any]] = []

        for topic_idx, topic in enumerate(self.model.components_):
            top_word_indices = topic.argsort()[:-8:-1]
            top_words = [
                {"word": str(feature_names[i]), "weight": round(float(topic[i]), 3)}
                for i in top_word_indices
            ]
            
            # Topic significance based on average document allocation
            topic_weight = float(doc_topic_dist[:, topic_idx].mean())

            topics_info.append({
                "topic_id": topic_idx + 1,
                "label": f"Cluster {topic_idx + 1}: {', '.join([w['word'] for w in top_words[:3]])}",
                "prevalence": round(topic_weight * 100, 1),
                "top_words": top_words
            })

        return topics_info
