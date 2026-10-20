# NeuroATS Intelligence

**Explainable resume analysis and job-description matching with Python, NLP, and machine learning.**

NeuroATS Intelligence is a Streamlit application that helps users inspect resume content, compare it with a target role, identify potential skill gaps, and review structured resume-improvement suggestions.

<p>
  <a href="https://neuroats-intelligence.streamlit.app/"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open live demo"></a>
  <a href="https://github.com/mightyalok00/NeuroATS-Intelligence/actions/workflows/ci.yml"><img src="https://github.com/mightyalok00/NeuroATS-Intelligence/actions/workflows/ci.yml/badge.svg" alt="GitHub Actions CI"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
  <img src="https://img.shields.io/badge/Python-3.11-blue.svg" alt="Python 3.11">
</p>

**Live demo:** [neuroats-intelligence.streamlit.app](https://neuroats-intelligence.streamlit.app/)

> **Responsible use:** Scores and rankings are decision-support signals, not hiring decisions, hiring probabilities, or validated predictors of job performance. Human review is required. See the project's [evaluation, fairness, and limitations guide](docs/EVALUATION_AND_FAIRNESS.md).

## What it does

- **Resume parsing:** extracts text and available sections from supported PDF, DOCX, and TXT documents.
- **Resume-quality feedback:** applies a configurable rubric to structure, contact details, action verbs, quantified achievements, skills, readability, and length.
- **Job-fit analysis:** combines lexical / semantic text similarity and detected skill overlap to highlight relevant content and potential gaps.
- **NLP tools:** includes TF-IDF, BM25, TextRank, and configurable embedding / LSA representations.
- **Career-document assistance:** suggests resume bullet revisions and drafts cover letters for user review.
- **Candidate comparison:** presents multiple profiles and role-fit signals in an interactive dashboard.
- **Interactive reporting:** displays results with Streamlit and Plotly.

## Architecture

~~~mermaid
flowchart TD
    A[PDF / DOCX / TXT] --> B[Text extraction and section parsing]
    B --> C[Preprocessing and feature engineering]
    C --> D[TF-IDF / BM25 / embeddings or LSA]
    C --> E[Resume-quality rubric]
    D --> F[Job matching and skill-gap analysis]
    D --> G[Ranking and extractive summarization]
    E --> H[Streamlit dashboard]
    F --> H
    G --> H
    H --> I[Human review]
~~~

### Algorithm reference

See **[Algorithm Notes](docs/ALGORITHMS.md)** for readable explanations of BM25, TF-IDF and cosine similarity, TextRank, embedding / LSA approaches, and the scoring rubric. The notes distinguish algorithm descriptions from empirical performance claims.

## Technology

- **Language / UI:** Python, Streamlit
- **ML / NLP:** scikit-learn, NumPy, SciPy, NLTK, BM25, TF-IDF, TextRank
- **Documents:** PDF and DOCX parsing utilities
- **Visualization:** Plotly
- **Testing / automation:** unittest and GitHub Actions

Dependencies are listed in [requirements.txt](requirements.txt). CI is configured for Python 3.11.

## Run locally

### Requirements

- Git
- Python 3.11
- pip

### Install

~~~bash
git clone https://github.com/mightyalok00/NeuroATS-Intelligence.git
cd NeuroATS-Intelligence
python -m venv .venv
~~~

Activate the environment:

**Windows PowerShell**
~~~powershell
.\.venv\Scripts\Activate.ps1
~~~

**macOS / Linux**
~~~bash
source .venv/bin/activate
~~~

Install dependencies and launch:

~~~bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
~~~

Open the local URL shown by Streamlit, usually http://localhost:8501.

## Test and benchmark

Run the unit tests:

~~~bash
python -m unittest discover -s tests -p "test_*.py" -v
~~~

Run the reproducible BM25 ranking smoke benchmark:

~~~bash
python -m benchmarks.evaluate_ranking
~~~

The benchmark uses a small, manually labelled illustrative corpus to verify ranking-metric calculation and provide a reproducible baseline for the lexical retrieval component. It is **not** a real-world resume-ranking benchmark and must not be presented as evidence of hiring accuracy, fairness, or generalization.

For an **authorized external dataset**, the optional harness compares keyword overlap, BM25, and TF-IDF and reports MRR, Recall@k, NDCG@k, and per-query misses. It expects a CSV with columns `query_id, query_text, document_id, candidate_id, document_text, relevance, split`, where `relevance` is `0` or `1` and `split` is `train`, `dev`, or `test`. It rejects candidate IDs appearing in multiple splits and duplicate query/document pairs.

```bash
python -m benchmarks.evaluate_external_dataset --input /path/to/authorized_dataset.csv --k 5 --output reports/external_ranking_evaluation.md
```

The script does not download or supply a dataset. Obtain data-use permission first, keep private data out of Git and CI, and follow [the external evaluation protocol](docs/REAL_WORLD_EVALUATION.md). The [blinded annotation worksheet](docs/ANNOTATION_WORKSHEET.csv) is a blank template, not completed human review. See the [fairness audit template](docs/FAIRNESS_AUDIT_TEMPLATE.md) and [operational validation checklist](docs/OPERATIONAL_VALIDATION.md). These plans do not establish real-world accuracy, fairness, or production readiness.

See the [CI workflow](https://github.com/mightyalok00/NeuroATS-Intelligence/actions/workflows/ci.yml) for automated checks.

## Evaluation, fairness, and privacy

Read [Evaluation, Fairness, and Limitations](docs/EVALUATION_AND_FAIRNESS.md) before interpreting scores or using the tool in consequential contexts.

A meaningful real-world evaluation still requires permitted representative data, candidate-level leakage prevention, explicit relevance labels, simple baselines, appropriate ranking metrics (such as Recall@k and NDCG@k), error analysis, and reproducible reporting. The included smoke benchmark only checks the mechanics on a small illustrative corpus.

Privacy considerations:
- Do not commit real resumes, contact details, credentials, or private evaluation data.
- Treat uploaded resume text as untrusted input.
- Check model configuration and cache state: a pretrained embedding model may be downloaded on first use; do not assume every setup is fully offline.

## Repository structure

~~~text
.
├── app.py
├── requirements.txt
├── src/
│   ├── algorithms/
│   ├── data_pipeline/
│   ├── feature_engineering/
│   └── models/
├── tests/
├── benchmarks/
│   └── evaluate_ranking.py
├── docs/
│   ├── ALGORITHMS.md
│   └── EVALUATION_AND_FAIRNESS.md
└── .github/workflows/ci.yml
~~~

Directory contents may evolve; consult the repository for the current implementation.

## Limitations

- The ATS-style score is a heuristic resume-feedback rubric, not a commercial ATS score or hiring probability.
- Similarity can miss paraphrases or overvalue shared vocabulary; detected skills depend on the taxonomy.
- PDF layout, columns, scanned documents, and unusual formatting can reduce extraction quality.
- Seniority estimates and generated text may be inaccurate; verify all suggested statements and never include invented qualifications or metrics.
- Unit tests and the illustrative smoke benchmark do not establish real-world validity, fairness, or production readiness.

## Roadmap

- Evaluate on a documented, permitted, held-out resume/job relevance dataset.
- Compare BM25 and current job-fit scoring against keyword and other simple baselines.
- Report Recall@k, NDCG@k, confidence intervals where feasible, and representative failure cases.
- Add parser fixtures for diverse resume layouts and document model-download / offline behavior.
- Strengthen privacy and fairness review before any consequential deployment.

## Contributing

1. Create a focused branch.
2. Add tests for changed behavior.
3. Run the test suite and ranking smoke benchmark.
4. Document changes to scoring, model behavior, data handling, or dependencies.
5. Open a pull request with the validation evidence and known limitations.

Do not commit real resumes, personal data, secrets, or private benchmark records.

## License

See [LICENSE](LICENSE) for the project's license terms.
