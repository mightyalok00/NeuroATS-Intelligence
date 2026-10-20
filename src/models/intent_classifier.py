"""
Adaptive ML Query Intent Classifier
Dynamically classifies user inquiries into distinct operational tasks using trained ML.
"""

import os
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import INTENT_CLASSES

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "intent_classifier.joblib")


class AdaptiveIntentClassifier:
    """Trained NLP classifier that routes queries without needing external LLMs."""

    def __init__(self, force_retrain: bool = False):
        self.intents = INTENT_CLASSES
        self.pipeline: Pipeline | None = None
        if not force_retrain and os.path.exists(MODEL_PATH):
            try:
                self.pipeline = joblib.load(MODEL_PATH)
            except (OSError, ValueError, EOFError):
                self._train_and_serialize()
        else:
            self._train_and_serialize()

    def _train_and_serialize(self):
        """Train classifier with diverse user queries across intent types and persist."""
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        dataset = [
            # SPECIFIC_DETAIL
            ("What programming languages does the candidate know?", "SPECIFIC_DETAIL"),
            ("What companies has he worked at?", "SPECIFIC_DETAIL"),
            ("Where did they go to university or get their degree?", "SPECIFIC_DETAIL"),
            ("List all the databases mentioned in the resume", "SPECIFIC_DETAIL"),
            ("Does the candidate have experience with Kubernetes or Docker?", "SPECIFIC_DETAIL"),
            ("What did they accomplish at their previous job?", "SPECIFIC_DETAIL"),
            ("What is the candidate's phone number or email address?", "SPECIFIC_DETAIL"),
            ("Show me their certifications or licenses", "SPECIFIC_DETAIL"),
            ("What projects did they build?", "SPECIFIC_DETAIL"),
            ("Did they use PyTorch or TensorFlow?", "SPECIFIC_DETAIL"),
            ("How many years of experience does this person have?", "SPECIFIC_DETAIL"),

            # ATS_DIAGNOSTICS
            ("What is my ATS score?", "ATS_DIAGNOSTICS"),
            ("How can I improve my resume for applicant tracking systems?", "ATS_DIAGNOSTICS"),
            ("Are there formatting errors or weak action verbs?", "ATS_DIAGNOSTICS"),
            ("Why did my resume get a low score?", "ATS_DIAGNOSTICS"),
            ("Critique my resume structure and readability", "ATS_DIAGNOSTICS"),
            ("What are the biggest weaknesses in this resume?", "ATS_DIAGNOSTICS"),
            ("Did I use enough quantifiable metrics and numbers?", "ATS_DIAGNOSTICS"),
            ("Give me suggestions to boost my score to 100", "ATS_DIAGNOSTICS"),

            # JOB_FIT_MATCH
            ("Does this resume match a Senior Machine Learning Engineer position?", "JOB_FIT_MATCH"),
            ("Compare this candidate to the job description", "JOB_FIT_MATCH"),
            ("What skills am I missing for this job?", "JOB_FIT_MATCH"),
            ("How well do I fit the requirements for this role?", "JOB_FIT_MATCH"),
            ("What is the match percentage for a Backend Developer?", "JOB_FIT_MATCH"),
            ("Is this candidate qualified for a Staff DevOps Engineer job?", "JOB_FIT_MATCH"),
            ("Highlight the gaps between my experience and the job posting", "JOB_FIT_MATCH"),

            # INTERVIEW_PREP
            ("Generate technical interview questions for this candidate", "INTERVIEW_PREP"),
            ("What behavioral interview questions should I ask based on their projects?", "INTERVIEW_PREP"),
            ("Give me practice questions for an upcoming interview", "INTERVIEW_PREP"),
            ("How can the candidate prepare for questions about their distributed systems project?", "INTERVIEW_PREP"),
            ("What are potential interview red flags or deep-dive topics?", "INTERVIEW_PREP"),

            # EXECUTIVE_SUMMARY
            ("Summarize this candidate's career trajectory", "EXECUTIVE_SUMMARY"),
            ("Give me a 30-second elevator pitch for this resume", "EXECUTIVE_SUMMARY"),
            ("What is the executive overview of this person's background?", "EXECUTIVE_SUMMARY"),
            ("Give me an executive summary highlighting top achievements", "EXECUTIVE_SUMMARY"),
            ("Briefly explain who this candidate is and what they specialize in", "EXECUTIVE_SUMMARY"),
        ]

        texts = [q[0] for q in dataset]
        labels = [q[1] for q in dataset]

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                stop_words="english",
                max_features=800
            )),
            ("clf", LogisticRegression(
                C=3.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            ))
        ])

        self.pipeline.fit(texts, labels)
        try:
            joblib.dump(self.pipeline, MODEL_PATH)
        except OSError:
            pass

    def classify_intent(self, query: str) -> dict[str, Any]:
        """Classify user query into one of the operational intents with confidence."""
        if self.pipeline is None:
            self._train_and_serialize()
        probs = self.pipeline.predict_proba([query])[0]
        classes = self.pipeline.classes_
        
        sorted_pairs = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
        top_intent = sorted_pairs[0][0]
        confidence = round(float(sorted_pairs[0][1]) * 100, 1)

        return {
            "intent": top_intent,
            "confidence": confidence,
            "all_scores": {c: round(float(p) * 100, 1) for c, p in sorted_pairs}
        }
