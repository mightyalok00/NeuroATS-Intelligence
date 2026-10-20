"""
Job Description Matcher and Skill Gap Analysis Engine
Combines dense semantic similarity, TF-IDF lexical match, and skill taxonomy overlap.
"""

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data_pipeline.skill_taxonomy import extract_skills_from_text
from src.feature_engineering.embeddings import DenseEmbedder


class JobFitMatcher:
    """Computes multidimensional candidate-to-job fit and identifies critical skill gaps."""

    def __init__(self, embedder: DenseEmbedder):
        self.embedder = embedder

    def evaluate_fit(self, resume_text: str, job_text: str) -> dict[str, Any]:
        """
        Evaluate candidate fit against a job description.
        Returns:
            composite_score: float (0-100)
            semantic_score: float (0-100)
            lexical_score: float (0-100)
            skill_match_ratio: float (0-100)
            matched_skills: list
            missing_skills: list
            candidate_bonus_skills: list
            priority_missing_keywords: list
        """
        if not resume_text.strip() or not job_text.strip():
            return {
                "composite_score": 0.0,
                "semantic_score": 0.0,
                "lexical_score": 0.0,
                "skill_match_ratio": 0.0,
                "matched_skills": [],
                "missing_skills": [],
                "candidate_bonus_skills": [],
                "priority_missing_keywords": []
            }

        # 1. Dense Semantic Similarity
        resume_dense = self.embedder.encode([resume_text])[0]
        job_dense = self.embedder.encode([job_text])[0]
        semantic_sim = max(0.0, min(1.0, self.embedder.cosine_similarity(resume_dense, job_dense)))
        semantic_score = round(semantic_sim * 100, 1)

        # 2. TF-IDF Lexical Similarity
        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=1200)
        tfidf_mat = vec.fit_transform([resume_text, job_text])
        lexical_sim = max(0.0, min(1.0, float(cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:2])[0][0])))
        lexical_score = round(lexical_sim * 100, 1)

        # 3. Hard & Soft Skill Overlap
        resume_skills_dict = extract_skills_from_text(resume_text)
        job_skills_dict = extract_skills_from_text(job_text)

        resume_skills_flat: set[str] = {
            s.lower() for cat_skills in resume_skills_dict.values() for s in cat_skills
        }
        job_skills_flat: set[str] = {
            s.lower() for cat_skills in job_skills_dict.values() for s in cat_skills
        }

        matched_skills = sorted([s.title() for s in (resume_skills_flat & job_skills_flat)])
        missing_skills = sorted([s.title() for s in (job_skills_flat - resume_skills_flat)])
        candidate_bonus_skills = sorted([s.title() for s in (resume_skills_flat - job_skills_flat)])

        total_job_skills = len(job_skills_flat)
        skill_match_ratio = (
            round((len(matched_skills) / total_job_skills) * 100, 1)
            if total_job_skills > 0 else 85.0
        )

        # 4. Composite Fit Score (Weighted blend of Semantic, Lexical, and Hard Skills)
        # 45% Hard Skill Match, 35% Dense Semantic, 20% Lexical TF-IDF
        composite = (0.45 * skill_match_ratio) + (0.35 * semantic_score) + (0.20 * lexical_score)
        composite = round(max(0.0, min(100.0, composite)), 1)

        # 5. Extract top TF-IDF keywords from Job Description that are absent from Resume
        job_terms = vec.get_feature_names_out()
        job_weights = tfidf_mat[1].toarray()[0]
        resume_words = set(resume_text.lower().split())

        missing_keywords = []
        sorted_job_term_indices = np.argsort(job_weights)[::-1]
        for idx in sorted_job_term_indices:
            term = str(job_terms[idx])
            if (
                term not in resume_words
                and not any(term in s.lower() for s in matched_skills)
                and len(term) > 3
                and not term.isdigit()
            ):
                missing_keywords.append(term)
            if len(missing_keywords) >= 8:
                break

        return {
            "composite_score": composite,
            "semantic_score": semantic_score,
            "lexical_score": lexical_score,
            "skill_match_ratio": skill_match_ratio,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "candidate_bonus_skills": candidate_bonus_skills,
            "priority_missing_keywords": missing_keywords
        }
