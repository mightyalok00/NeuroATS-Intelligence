"""
Supervised Domain Classification Model
Classifies resumes into specialized engineering/tech domains using calibrated ML models.
"""

import os
from typing import Dict, List, Tuple
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.config import DOMAINS

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "domain_classifier.joblib")


class DomainClassifier:
    """Trained machine learning classifier for predicting resume professional domains."""

    def __init__(self, force_retrain: bool = False):
        self.domains = DOMAINS
        self.model: Pipeline = None
        if not force_retrain and os.path.exists(MODEL_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
            except Exception:
                self._train_and_serialize()
        else:
            self._train_and_serialize()

    def _train_and_serialize(self):
        """Train classifier with curated domain profiles and serialize to disk."""
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        training_data = [
            # Machine Learning & AI
            ("Machine learning deep learning pytorch tensorflow transformers nlp computer vision llm model training evaluation scikit-learn neural networks huggingface gpu cuda hyperparameter tuning feature engineering mlops wandb mlflow embeddings rag", "Machine Learning & AI"),
            ("AI research scientist developing diffusion models transformers generative AI reinforcement learning pytorch jax math statistics python deep neural networks", "Machine Learning & AI"),
            ("MLOps engineer deployed LLM serving pipelines using vLLM Triton TensorRT Triton inference server Docker Kubernetes CI/CD Prometheus MLflow", "Machine Learning & AI"),
            ("Data Scientist built predictive models random forest xgboost customer churn lifetime value A/B testing python pandas numpy statsmodels regression", "Machine Learning & AI"),

            # Full-Stack & Backend
            ("Backend engineer architected high throughput microservices using Go Python FastAPI PostgreSQL Redis Kafka Docker gRPC REST API system design distributed systems", "Full-Stack & Backend"),
            ("Full-stack developer building scalable web applications with React TypeScript Node.js Express MongoDB PostgreSQL GraphQL RESTful APIs Tailwind CSS Next.js", "Full-Stack & Backend"),
            ("Senior software engineer Java Spring Boot Hibernate MySQL RabbitMQ microservices architecture AWS cloud REST APIs unit testing JUnit mockito", "Full-Stack & Backend"),
            ("Django Python developer built database ORM caching Redis Celery background workers Postgres authentication JWT OAuth2", "Full-Stack & Backend"),

            # DevOps & Cloud Infrastructure
            ("DevOps cloud engineer managing AWS infrastructure EC2 S3 Lambda ECS EKS Terraform Ansible Docker Kubernetes Helm CI/CD GitHub Actions Jenkins Prometheus Grafana", "DevOps & Cloud Infrastructure"),
            ("Site reliability engineer SRE automated cloud infrastructure with Terraform multi-region Kubernetes cluster monitoring alerts Datadog Linux networking bash", "DevOps & Cloud Infrastructure"),
            ("Cloud architect designing GCP infrastructure Google Cloud Platform BigQuery IAM VPC Cloud Run Serverless container security automated deployment pipelines", "DevOps & Cloud Infrastructure"),

            # Data Engineering & Analytics
            ("Data engineer built large scale ETL ELT data pipelines using Apache Spark PySpark Kafka Airflow dbt Snowflake Databricks data lakehouse SQL warehousing", "Data Engineering & Analytics"),
            ("Big data developer managing Hadoop HDFS Spark streaming Kafka clusters Hive Presto Trino parquet data pipeline ingestion orchestration", "Data Engineering & Analytics"),
            ("BI and analytics engineer creating Tableau dashboards Power BI SQL data modeling ETL reporting Snowflake KPIs stakeholder analytics", "Data Engineering & Analytics"),

            # Frontend & Mobile
            ("Frontend engineer specialized in React Next.js TypeScript Redux Zustand Tailwind CSS HTML5 CSS3 responsive design Webpack Vite UI/UX web performance", "Frontend & Mobile"),
            ("Mobile app developer built iOS and Android applications with Flutter Dart React Native Swift Kotlin mobile UI state management App Store deployment", "Frontend & Mobile"),
            ("Vue.js Nuxt frontend engineer implemented component libraries design systems CSS animations accessibility performance optimization", "Frontend & Mobile"),

            # Cybersecurity & Systems
            ("Cybersecurity analyst security engineer vulnerability assessment penetration testing SIEM Wireshark IAM OAuth2 Zero Trust network security cryptography", "Cybersecurity & Systems"),
            ("Systems security and compliance SOC2 ISO27001 GDPR audit threat intelligence firewall Linux hardening incident response security monitoring", "Cybersecurity & Systems"),

            # Product & Technical Management
            ("Technical product manager TPM led agile sprint planning roadmap feature prioritization stakeholder management user stories JIRA scrum OKRs KPI delivery", "Product & Technical Management"),
            ("Engineering manager led cross-functional team of 15 engineers sprint cycles hiring 1-on-1 mentorship delivery architecture alignment budgeting", "Product & Technical Management"),
        ]

        texts = [item[0] for item in training_data]
        labels = [item[1] for item in training_data]

        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                max_features=1500,
                stop_words="english"
            )),
            ("clf", LogisticRegression(
                C=2.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            ))
        ])

        self.model.fit(texts, labels)
        try:
            joblib.dump(self.model, MODEL_PATH)
        except Exception:
            pass

    def predict(self, text: str) -> Dict[str, any]:
        """
        Predict domain and return full probability distribution.
        """
        probs = self.model.predict_proba([text])[0]
        classes = self.model.classes_

        # Sort by probability descending
        sorted_pairs = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
        top_domain = sorted_pairs[0][0]
        top_confidence = round(float(sorted_pairs[0][1]) * 100, 1)

        breakdown = {
            domain: round(float(prob) * 100, 1)
            for domain, prob in sorted_pairs
        }

        return {
            "predicted_domain": top_domain,
            "confidence": top_confidence,
            "distribution": breakdown
        }
