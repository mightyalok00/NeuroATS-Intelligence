# NeuroATS Intelligence

**An explainable, machine-learning-powered resume analysis and job-fit platform.**

NeuroATS Intelligence helps candidates and recruiters inspect resume structure, compare resume content with job requirements, identify skill gaps, and produce evidence-oriented feedback. It combines NLP features and classical machine-learning methods in a Streamlit interface.

<p align="left">
  <a href="https://neuroats-intelligence.streamlit.app/"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open live demo"></a>
  <a href="https://github.com/mightyalok00/NeuroATS-Intelligence/actions/workflows/ci.yml"><img src="https://github.com/mightyalok00/NeuroATS-Intelligence/actions/workflows/ci.yml/badge.svg" alt="CI status"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
  <img src="https://img.shields.io/badge/Python-3.11-blue.svg" alt="Python 3.11">
  <img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/ML-scikit--learn-F7931E.svg" alt="scikit-learn">
</p>

> **Live demo:** [neuroats-intelligence.streamlit.app](https://neuroats-intelligence.streamlit.app/)

> **Project status:** Portfolio / research-oriented application. Scores and rankings are decision-support signals, not hiring decisions or validated predictors of job performance. Review the limitations and evaluation protocol before using the output in consequential settings.

## Contents

- [Overview](#overview)
- [Key capabilities](#key-capabilities)
- [How it works](#how-it-works)
- [Algorithm reference](#algorithm-reference)
- [Technology stack](#technology-stack)
- [Getting started](#getting-started)
- [Tests](#tests)
- [Evaluation, fairness, and privacy](#evaluation-fairness-and-privacy)
- [Repository structure](#repository-structure)
- [Known limitations](#known-limitations)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

## Overview

Resume screening can be difficult to audit when a system provides only a single score. NeuroATS aims to make resume analysis more inspectable by combining structured parsing, explicit scoring dimensions, job-description comparison, and supporting evidence.

The application is designed to help users:
- Review resume sections and extracted contact information.
- Compare resume skills and experience with a target job description.
- Inspect missing or underrepresented keywords and skills.
- Explore resume readability, action-verb use, and quantified impact.
- Draft resume bullet improvements and cover letters for human review.
- Compare multiple candidate profiles in a dashboard.

The project uses a local-first architecture for much of its text processing. Some embedding configurations may attempt to load a pretrained model on first use; see [Known limitations](#known-limitations).

## Key capabilities

| Capability | What it does |
|---|---|
| Resume parsing | Reads supported PDF, DOCX, and TXT files and extracts available resume sections and contact details. Extraction quality depends on document layout and text quality. |
| ATS-style rubric | Calculates a deterministic, multi-dimensional heuristic score across configured resume-quality criteria. It is not an estimate of any particular employer's proprietary ATS score. |
| Job-fit analysis | Compares resume content with a job description using text similarity and skill overlap, then highlights potential gaps. |
| Skill taxonomy | Matches resume terms against a curated technical-skills taxonomy. Coverage and terminology may be incomplete. |
| NLP analysis | Uses TF-IDF, BM25, text-ranking, and other classical NLP / ML techniques for retrieval, analysis, and summarization tasks. |
| Candidate comparison | Supports batch comparison and a dashboard view for inspecting candidate profiles side by side. |
| Resume tailoring | Suggests wording improvements and can export generated documents for the user to review. Users should verify every statement for accuracy. |
| Cover-letter drafting | Produces a structured draft based on the supplied resume and role context. |
| Career insights | Provides an ML-based seniority estimate and suggested development areas; these are indicative, not validated career predictions. |
| Interactive dashboard | Presents analysis and comparison outputs through Streamlit and Plotly visualizations. |

## How it works

The pipeline is organized into five logical stages:

1. **Ingestion** — accept a supported resume file and extract text and sections.
2. **Preprocessing** — normalize text and derive features such as action verbs, metrics, and readability indicators.
3. **Representation** — create lexical representations with TF-IDF and BM25; use dense embeddings or an LSA/SVD fallback where configured.
4. **Analysis** — run configured classification, topic modeling, similarity, ranking, skill matching, and heuristic scoring components.
5. **Presentation** — display results and supporting evidence in the Streamlit dashboard.

~~~mermaid
flowchart TD
    A[PDF / DOCX / TXT resume] --> B[Text extraction and parsing]
    B --> C[Cleaning and feature engineering]
    C --> D[TF-IDF / BM25 / embeddings or LSA]
    D --> E[Job fit and skill-gap analysis]
    C --> F[Resume-quality rubric]
    D --> G[Ranking, classification and summarization]
    E --> H[Streamlit dashboard]
    F --> H
    G --> H
    H --> I[Human review and final decision]
~~~

### Scoring and interpretability

The ATS-style score is a configurable, deterministic rubric with weighted dimensions such as contact details, structure, action verbs, quantified impact, technical skills, readability, and length. It should be interpreted as **resume feedback**, not as a hiring probability or a guarantee of compatibility with a commercial applicant-tracking system.

Job-fit outputs combine text similarity and skill overlap. They can be affected by synonyms, missing context, formatting, and the completeness of the skill taxonomy. Where available, inspect the matched terms and evidence rather than relying on a single aggregate number.

## Algorithm reference

The implementation uses established techniques for lexical retrieval, text similarity, sentence ranking, and configurable resume-quality scoring. For readable formulas, assumptions, interpretation guidance, and evaluation caveats, see **[Algorithm Notes](docs/ALGORITHMS.md)**. The equations describe general algorithm families; they should not be interpreted as benchmark results or proof of model performance.

## Technology stack

- **Language:** Python
- **Application:** Streamlit
- **Machine learning / NLP:** scikit-learn, NumPy, SciPy, NLTK and related project dependencies
- **Text and document processing:** PDF and DOCX parsing utilities
- **Visualization:** Plotly
- **Model artifacts:** joblib
- **Automation:** GitHub Actions and Python unit tests

The exact dependency versions are defined in [requirements.txt](requirements.txt). The CI workflow is configured for Python 3.11.

## Getting started

### Prerequisites

- Git
- Python 3.11 recommended
- pip

### 1. Clone the repository

~~~bash
git clone https://github.com/mightyalok00/NeuroATS-Intelligence.git
cd NeuroATS-Intelligence
~~~

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
~~~

**macOS / Linux:**

~~~bash
python3.11 -m venv .venv
source .venv/bin/activate
~~~

### 3. Install dependencies

~~~bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

### 4. Start the application

~~~bash
streamlit run app.py
~~~

Open the local URL printed by Streamlit (usually http://localhost:8501).

### 5. Run tests

From the repository root:

~~~bash
python -m unittest discover -s tests -p "test_*.py" -v
~~~

The same discovery command is configured in GitHub Actions. Check the [CI workflow](https://github.com/mightyalok00/NeuroATS-Intelligence/actions/workflows/ci.yml) for the latest run result.

## Evaluation, fairness, and privacy

See **[Evaluation, Fairness, and Limitations](docs/EVALUATION_AND_FAIRNESS.md)** for the evaluation protocol, reporting expectations, fairness considerations, privacy guidance, and release checklist.

Before drawing performance conclusions, evaluation should use representative, held-out data and document:
- Dataset provenance, consent and permitted use.
- Train / validation / test separation and leakage controls.
- Baselines, per-class metrics where relevant, and error analysis.
- Subgroup performance and limitations where lawful and appropriate.
- Reproducible configurations and known failure cases.

This repository's evaluation guidance describes what should be measured; it should not be mistaken for evidence that those benchmarks have already been run. Do not claim accuracy, fairness, or production-readiness results unless they are backed by published, reproducible measurements.

## Repository structure

~~~text
.
├── app.py
├── requirements.txt
├── README.md
├── docs/
│   └── EVALUATION_AND_FAIRNESS.md
├── src/
│   ├── algorithms/
│   ├── data_pipeline/
│   ├── feature_engineering/
│   ├── models/
│   ├── ui/
│   ├── artifacts/
│   ├── config.py
│   └── sample_data.py
├── tests/
│   └── test_pipeline.py
└── .github/
    └── workflows/
        └── ci.yml
~~~

Directory contents may evolve as the project develops; consult the repository for the current implementation.

## Known limitations

- **Not a hiring decision system:** The application is intended for exploratory resume feedback and decision support. A human should review outputs before taking action.
- **Heuristic score:** The rubric is designed for consistency and feedback, not to reproduce proprietary ATS behavior or predict candidate success.
- **Data and domain coverage:** A curated skill taxonomy and model training data cannot cover every role, industry, language, career path, or nontraditional background.
- **Parsing errors:** Columns, tables, unusual fonts, scans, and complex PDF layouts can cause incomplete or incorrect extraction.
- **Embedding model availability:** Depending on configuration and local cache state, a pretrained sentence-transformer model may need to be downloaded on first use. The LSA/SVD fallback may have different behavior. Do not assume every configuration is fully offline on a fresh machine.
- **No published benchmark claim:** CI and unit tests check software behavior; they do not by themselves establish real-world ranking validity, fairness, or generalization.
- **Generated text requires review:** Resume and cover-letter suggestions may be incomplete or unsuitable. Do not include invented experience, skills, qualifications, or metrics.

## Roadmap

Potential next steps for strengthening the project:

- Publish a reproducible benchmark with a documented, held-out evaluation set.
- Add more parser fixtures and edge-case tests for resumes with varied layouts.
- Report component-level metrics, failure analysis, and confidence intervals where appropriate.
- Evaluate subgroup outcomes and document mitigation decisions.
- Improve model download / cache controls and make offline behavior explicit.
- Add privacy-conscious data handling guidance and operational safeguards for deployments.

These are development goals, not claims that the capabilities are already complete.

## Contributing

Contributions are welcome. For a change:

1. Create a focused branch.
2. Add or update tests for changed behavior.
3. Run the test suite locally.
4. Document changes to scoring, model behavior, data handling, or dependencies.
5. Open a pull request with a clear summary, testing evidence, and known limitations.

Please avoid committing real resumes, personal data, credentials, or other sensitive files.

## License

Distributed under the [MIT License](LICENSE). If the repository does not yet include a LICENSE file, add one before relying on the license badge or redistribution terms.
