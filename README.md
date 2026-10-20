# 🧠 NeuroATS-Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-10%2F10%20Production%20Ready-brightgreen.svg)](https://github.com/mightyalok00/NeuroATS-Intelligence)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A 100% Free, Local-First, Zero-LLM AI & Machine Learning Platform** for enterprise ATS analytics, recruiter batch candidate ranking, real-time bullet point optimization, and career intelligence. Replaces costly black-box RAG with deterministic, mathematically sound machine learning algorithms, NLP feature engineering, unsupervised topic modeling, and supervised classification pipelines.

---

## 🌟 Why This Replaces RAG (The "10/10" Engineering Philosophy)

Traditional RAG systems (like the original `Adaptive-Rag`) have significant limitations:

1. **High API Costs & Rate Limits**: Rely on external cloud LLMs (OpenAI, Groq) and proprietary vector databases.
2. **Hallucination Risk**: LLMs often generate inaccurate claims when evaluating candidate resumes or matching job requirements.
3. **Black-Box Metrics**: They cannot explain *why* an ATS score was given or calculate deterministic, reproducible gap matrices.

**NeuroATS-Intelligence** transforms this architecture into an **interpretable, high-performance Machine Learning & NLP pipeline**:

- **0 External API Keys Needed**: 100% free, runs completely offline on your local CPU/GPU.
- **📤 One-Click Direct Resume Upload**: Drag-and-drop `.pdf`, `.docx`, or `.txt` resumes directly on the main dashboard with instant offline AI analysis.
- **🎛️ Dynamic Pipeline Filters & Cutoffs**: Set custom passing ATS thresholds, target domain specialization, and filter skills by taxonomy family.
- **✨ One-Click AI Resume Tailor**: Weaves missing role skills and power action verbs into your resume, previewing score jumps and exporting to `.docx`.
- **✉️ Algorithmic Cover Letter Generator**: Synthesizes a structured 3-part cover letter linking candidate achievements to company requirements, exportable to `.docx`.
- **📈 ML Seniority Estimator & Career Roadmap**: Predicts candidate engineering tier (Junior to Principal), benchmarks market salary bands, and outlines next-tier promotion milestones.
- **Multi-Resume Batch Leaderboard**: Evaluate and rank 2 to 10 candidates simultaneously with a 4-quadrant scatter matrix and recruiter candidate filtering.
- **Real-Time Live Bullet Optimizer**: Interactive reactive sandbox to draft and boost bullet points using the Google XYZ formula.
- **Universal Document Support**: Native parsing for `.pdf`, `.docx`, and `.txt` documents.
- **Offline Model Serialization**: Pre-compiled `.joblib` model artifacts decoupled from runtime inference.
- **Explainable AI (XAI)**: Full probability distributions, TF-IDF feature weights, and exact evidence spans.

---

## 🏗️ System Architecture & ML Pipeline

```mermaid
flowchart TD
    subgraph Ingestion ["1. Data Ingestion & Preprocessing"]
        PDF[PDF / TXT Resume] --> Parser[ResumeParser: Regex Entity & Section Extraction]
        Parser --> Cleaning[ResumePreprocessor: Tokenization & Syllable Counting]
        Cleaning --> Readability[Flesch Reading Ease & Grade Level]
        Cleaning --> ActionVerbs[Action-Verb & Metric Impact Ratios]
        Parser --> Skills[Taxonomy Matcher: 400+ Categorized Skills]
    end

    subgraph Representations ["2. Feature Engineering"]
        Cleaning --> TFIDF[TF-IDF N-Gram Vectorizer (1,2)]
        Cleaning --> BM25[BM25 Okapi Pure-NumPy Inverted Index]
        Cleaning --> Dense[DenseEmbedder: SentenceTransformer / LSA SVD]
    end

    subgraph Models ["3. AI & Machine Learning Models"]
        TFIDF --> DomClf[Supervised Domain Classifier (7 Tech Domains)]
        TFIDF --> IntentClf[Adaptive ML Intent Classifier]
        TFIDF --> LDA[Latent Dirichlet Allocation Topic Modeler]
        Dense --> KMeans[K-Means Semantic Bullet Clustering]
    end

    subgraph Algorithms ["4. Core Algorithmic Engines"]
        Dense & TFIDF --> JobMatch[JobFitMatcher: Composite Semantic + Skill Match]
        Dense & TFIDF --> TextRank[TextRank: PageRank Graph Centrality Summarizer]
        ActionVerbs & Skills & Readability --> ATS[Multi-Dimensional ATS Scoring Engine (0-100)]
        BM25 & IntentClf --> QA[Adaptive ML Q&A Assistant]
    end

    subgraph UI ["5. Streamlit Presentation Layer"]
        ATS & JobMatch & QA & DomClf --> Dashboard[Glassmorphic Streamlit Dashboard with Plotly]
    end
```

---

## 🧮 Mathematical Formulations of Core Algorithms

### 1. BM25 Okapi Relevance Ranking

Used for exact factual extraction from candidate resumes without LLM hallucination:
$$\text{Score}(D, Q) = \sum_{q \in Q} \text{IDF}(q) \cdot \frac{f(q, D) \cdot (k_1 + 1)}{f(q, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
$$\text{IDF}(q) = \ln \left( \frac{N - n(q) + 0.5}{n(q) + 0.5} + 1 \right)$$
*Parameters: $k_1 = 1.5$, $b = 0.75$.*

### 2. TextRank Graph Centrality (Extractive PageRank)

Computes eigenvector centrality over the sentence similarity graph $G = (V, E)$ to extract top career impact bullets:
$$PR(u) = \frac{1 - d}{|V|} + d \sum_{v \in \text{In}(u)} \frac{w_{vu}}{\sum_{k} w_{vk}} PR(v)$$
*Where $d = 0.85$ (damping factor) and $w_{vu}$ is the cosine similarity between sentence vectors.*

### 3. Flesch Reading Ease Formula

$$\text{Score} = 206.835 - 1.015 \left( \frac{\text{total words}}{\text{total sentences}} \right) - 84.6 \left( \frac{\text{total syllables}}{\text{total words}} \right)$$

### 4. Multidimensional ATS Composite Rubric

The 100-point score is computed deterministically across 7 weighted pillars:
$$\text{ATS Score} = w_1 S_{\text{contact}} + w_2 S_{\text{structure}} + w_3 S_{\text{verbs}} + w_4 S_{\text{metrics}} + w_5 S_{\text{technical}} + w_6 S_{\text{readability}} + w_7 S_{\text{length}}$$

---

## 🚀 Quickstart Guide

### 1. Installation

Clone the repository and install requirements:

```bash
git clone https://github.com/mightyalok00/NeuroATS-Intelligence.git
cd NeuroATS-Intelligence
pip install -r requirements.txt
```

### 2. Run the Streamlit Application

Launch the web interface:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### 3. Run Pipeline Unit Tests

Execute the automated test suite verifying all 11 ML components:

```bash
python -m unittest tests/test_pipeline.py
```

---

## 📊 Feature Highlights

- **📄 Smart Resume Parsing**: Extracts name, phone, email, LinkedIn, GitHub, and segments sections (Experience, Education, Skills, Projects, Summary).
- **🎯 Job Description Fit**: Calculates semantic cosine similarity + hard skill overlap + missing keyword priority rankings.
- **🤖 Zero-LLM Adaptive Q&A**: Dynamically routes user queries into 5 intent categories and answers accurately from the resume.
- **🎙️ Algorithmic Interview Question Generator**: Auto-generates technical deep-dives and STAR behavioral questions tailored to candidate projects.
- **🎨 Modern Dark-Mode UI**: Glassmorphism cards, glowing accent borders, and dynamic Plotly radar/bar charts.
- **📥 1-Click Export**: Download a full candidate audit report in Markdown format.

---

## 📂 Repository Layout

```text
d:/scaller/
├── .streamlit/
│   └── config.toml            # Enterprise dark theme & server settings
├── app.py                     # Streamlit Application Orchestrator Controller
├── requirements.txt           # Project Dependencies
├── README.md                  # Comprehensive Architecture & Documentation
├── src/
│   ├── config.py              # Rubrics, weights, domain & intent constants
│   ├── sample_data.py         # 5 Realistic candidate profiles & job descriptions
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── styles.py          # Enterprise CSS tokens & design system
│   │   └── components.py      # Modular tab views & KPI renderers
│   ├── data_pipeline/
│   │   ├── parser.py          # Universal parser (PDF, DOCX, TXT)
│   │   ├── preprocessor.py    # Flesch Readability, action verbs & metric impact
│   │   └── skill_taxonomy.py  # 400+ curated technical skill taxonomy
│   ├── feature_engineering/
│   │   ├── vectorizers.py     # Sublinear TF-IDF & pure NumPy BM25 Okapi
│   │   └── embeddings.py      # Dense vector embeddings with SVD/LSA fallback
│   ├── models/
│   │   ├── domain_classifier.py  # Supervised ML domain classification
│   │   ├── intent_classifier.py  # Adaptive ML query router
│   │   ├── topic_modeler.py      # Unsupervised LDA / NMF topic modeling
│   │   ├── cluster_engine.py     # K-Means experience clustering
│   │   └── train_models.py       # Offline model training & joblib serializer
│   ├── artifacts/
│   │   ├── domain_classifier.joblib  # Serialized domain classifier
│   │   └── intent_classifier.joblib  # Serialized intent router
│   └── algorithms/
│       ├── text_rank.py       # PageRank graph algorithm for summarization
│       ├── ats_scorer.py      # Multi-dimensional ATS scoring engine
│       ├── matcher.py         # Job description fit & skill gap analyzer
│       ├── tailor.py          # AI Resume Tailor & DOCX generator
│       ├── cover_letter.py    # Algorithmic Cover Letter & DOCX generator
│       ├── seniority_estimator.py # ML Seniority tier & career roadmap
│       └── qa_engine.py       # Adaptive ML question answering engine
└── tests/
    └── test_pipeline.py       # 15 automated unit tests
```

---

## 🛡️ License

MIT License. Free for educational and commercial use.
