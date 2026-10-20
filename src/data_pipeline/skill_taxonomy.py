"""
Curated Skill Taxonomy with Domain Mapping and Regex Patterns
"""

import re

SKILL_TAXONOMY: dict[str, list[str]] = {
    "Programming Languages": [
        "python", "java", "c++", "c#", "c", "golang", "go", "rust", "typescript",
        "javascript", "ruby", "php", "swift", "kotlin", "scala", "r", "julia",
        "bash", "shell", "powershell", "perl", "dart", "solidity", "sql"
    ],
    "Machine Learning & AI": [
        "machine learning", "deep learning", "neural networks", "nlp", "natural language processing",
        "computer vision", "llm", "large language models", "transformers", "bert", "gpt",
        "reinforcement learning", "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn",
        "xgboost", "lightgbm", "catboost", "hugging face", "huggingface", "spacy", "nltk",
        "opencv", "langchain", "langgraph", "llamaindex", "rag", "retrieval augmented generation",
        "fine-tuning", "prompt engineering", "feature engineering", "mlops", "wandb",
        "mlflow", "dvc", "bentoml", "onnx", "tensorrt", "triton", "cuda"
    ],
    "Data Engineering & Analytics": [
        "spark", "pyspark", "apache spark", "hadoop", "kafka", "apache kafka",
        "airflow", "apache airflow", "dbt", "snowflake", "bigquery", "databricks",
        "etl", "elt", "data pipeline", "data warehousing", "data modeling", "pandas",
        "numpy", "scipy", "polars", "tableau", "power bi", "looker", "redshift"
    ],
    "Cloud & DevOps": [
        "aws", "amazon web services", "azure", "gcp", "google cloud", "docker",
        "kubernetes", "k8s", "terraform", "ansible", "helm", "ci/cd", "github actions",
        "gitlab ci", "jenkins", "circleci", "prometheus", "grafana", "linux", "ubuntu",
        "nginx", "cloudformation", "serverless", "lambda", "ecs", "eks", "gke"
    ],
    "Databases & Storage": [
        "postgresql", "postgres", "mysql", "mongodb", "redis", "elasticsearch",
        "cassandra", "dynamodb", "sqlite", "neo4j", "qdrant", "pinecone", "chromadb",
        "milvus", "faiss", "weaviate"
    ],
    "Web & Backend": [
        "fastapi", "django", "flask", "node.js", "nodejs", "express", "spring boot",
        "graphql", "rest api", "restful", "grpc", "microservices", "websockets",
        "celery", "rabbitmq", "kafka"
    ],
    "Frontend & Mobile": [
        "react", "react.js", "next.js", "vue", "vue.js", "angular", "svelte",
        "html5", "css3", "tailwind", "tailwindcss", "sass", "bootstrap",
        "react native", "flutter", "ios", "android", "webpack", "vite"
    ],
    "Cybersecurity & Infrastructure": [
        "oauth2", "jwt", "ssl", "tls", "penetration testing", "vulnerability assessment",
        "siem", "wireshark", "cryptography", "zero trust", "iam", "soc2", "gdpr"
    ],
    "Software Architecture & Practices": [
        "system design", "distributed systems", "design patterns", "oop",
        "functional programming", "solid principles", "test driven development",
        "tdd", "unit testing", "pytest", "code review", "agile", "scrum", "kanban"
    ],
    "Soft Skills & Leadership": [
        "leadership", "mentorship", "cross-functional collaboration",
        "stakeholder management", "project management", "problem solving",
        "critical thinking", "communication", "adaptability", "team player"
    ]
}

# Inverted mapping: skill -> category
SKILL_TO_CATEGORY: dict[str, str] = {}
for category, skills in SKILL_TAXONOMY.items():
    for skill in skills:
        SKILL_TO_CATEGORY[skill.lower()] = category


def build_skill_pattern(skill: str) -> re.Pattern:
    """Compile regex pattern matching the skill as a whole token with punctuation tolerance."""
    escaped = re.escape(skill)
    # Special handling for skills with symbols like c++, c#, .js, etc.
    if skill in ["c++", "c#", "c", "r", "go"]:
        return re.compile(rf"(?<![a-zA-Z0-9]){escaped}(?![a-zA-Z0-9])", re.IGNORECASE)
    else:
        return re.compile(rf"\b{escaped}\b", re.IGNORECASE)


COMPILED_SKILL_PATTERNS = {
    skill: build_skill_pattern(skill)
    for category, skills in SKILL_TAXONOMY.items()
    for skill in skills
}


def extract_skills_from_text(text: str) -> dict[str, list[str]]:
    """
    Extract skills from text and group by category.
    Returns:
        dict of category -> list of matched skills
    """
    matched_by_cat: dict[str, set[str]] = {cat: set() for cat in SKILL_TAXONOMY}
    
    for skill, pattern in COMPILED_SKILL_PATTERNS.items():
        if pattern.search(text):
            category = SKILL_TO_CATEGORY.get(skill, "Other")
            matched_by_cat[category].add(skill.title() if len(skill) > 3 else skill.upper())
            
    # Clean up empty categories
    return {
        cat: sorted(skills)
        for cat, skills in matched_by_cat.items()
        if len(skills) > 0
    }
