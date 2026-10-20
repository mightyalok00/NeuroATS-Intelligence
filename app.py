"""
AdaptiveML Resume & Document Intelligence Platform
Production-Grade Streamlit Controller Architecture.

Author: Senior AI/ML Platform Engineer
Architecture: Non-RAG Machine Learning Pipelines & Modular UI Presentation Layer
"""

import streamlit as st

from src.algorithms.ats_scorer import ATSScorer
from src.algorithms.matcher import JobFitMatcher
from src.algorithms.qa_engine import AdaptiveQAEngine
from src.algorithms.seniority_estimator import SeniorityEstimator
from src.algorithms.text_rank import TextRankSummarizer
from src.config import APP_SUBTITLE, APP_TITLE, VERSION
from src.data_pipeline.parser import ResumeParser
from src.data_pipeline.preprocessor import ResumePreprocessor
from src.data_pipeline.skill_taxonomy import extract_skills_from_text
from src.feature_engineering.embeddings import DenseEmbedder
from src.feature_engineering.vectorizers import TFIDFEngine
from src.models.cluster_engine import ExperienceClusterEngine
from src.models.domain_classifier import DomainClassifier
from src.models.intent_classifier import AdaptiveIntentClassifier
from src.models.topic_modeler import TopicModeler
from src.sample_data import SAMPLE_JOB_DESCRIPTIONS, SAMPLE_RESUMES
from src.ui import (
    ENTERPRISE_CSS,
    render_header,
    render_kpis,
    render_tab_ats,
    render_tab_batch,
    render_tab_cover,
    render_tab_export,
    render_tab_match,
    render_tab_ml_explorer,
    render_tab_optimizer,
    render_tab_qa,
    render_tab_seniority,
    render_tab_tailor,
)

# ---------------------------------------------------------
# Page Configuration & CSS Injection
# ---------------------------------------------------------
st.set_page_config(
    page_title="NeuroATS Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(ENTERPRISE_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Singleton AI / ML Models Infrastructure
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Initializing AI & ML Pipelines...")
def load_ml_infrastructure():
    """Instantiate and cache all machine learning pipelines."""
    dense_embedder = DenseEmbedder(use_transformer=True)
    domain_clf = DomainClassifier()
    intent_clf = AdaptiveIntentClassifier()
    matcher = JobFitMatcher(dense_embedder)
    summarizer = TextRankSummarizer()
    qa_engine = AdaptiveQAEngine(
        intent_classifier=intent_clf,
        domain_classifier=domain_clf,
        embedder=dense_embedder,
        matcher=matcher,
        summarizer=summarizer
    )
    cluster_engine = ExperienceClusterEngine(dense_embedder)
    topic_modeler = TopicModeler(n_topics=3, method="lda")
    tfidf_engine = TFIDFEngine(max_features=1000)

    return {
        "embedder": dense_embedder,
        "domain_clf": domain_clf,
        "intent_clf": intent_clf,
        "matcher": matcher,
        "summarizer": summarizer,
        "qa_engine": qa_engine,
        "cluster_engine": cluster_engine,
        "topic_modeler": topic_modeler,
        "tfidf_engine": tfidf_engine
    }

ml_stack = load_ml_infrastructure()


# ---------------------------------------------------------
# Sidebar Controls & Document Ingestion
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 📂 Input Selection")
    
    input_mode = st.radio(
        "Source Method:",
        ["⚡ Preloaded High-Quality Sample", "📤 Upload Custom PDF / DOCX / TXT", "✍️ Paste Resume Text"],
        index=0
    )

    resume_text = ""
    candidate_file_name = "Sample Resume"

    if input_mode == "⚡ Preloaded High-Quality Sample":
        chosen_sample = st.selectbox(
            "Select Candidate Profile:",
            list(SAMPLE_RESUMES.keys()),
            index=0
        )
        resume_text = SAMPLE_RESUMES[chosen_sample]
        candidate_file_name = chosen_sample

    elif input_mode == "📤 Upload Custom PDF / DOCX / TXT":
        uploaded_file = st.file_uploader(
            "Upload Resume (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            help="Your file is parsed 100% locally on your machine with zero cloud upload."
        )
        if uploaded_file is not None:
            candidate_file_name = uploaded_file.name
            resume_text = ResumeParser.parse_file(uploaded_file.name, uploaded_file.read())
        else:
            st.info("Upload a file or switch to 'Preloaded Sample' to explore.")
            resume_text = SAMPLE_RESUMES[next(iter(SAMPLE_RESUMES.keys()))]

    else:
        pasted = st.text_area("Paste Resume Text Here:", height=240)
        resume_text = pasted if pasted.strip() else SAMPLE_RESUMES[next(iter(SAMPLE_RESUMES.keys()))]

    st.markdown("---")
    st.markdown("### 🎯 Target Job Role")
    job_mode = st.radio(
        "Job Description:",
        ["Standard Tech Roles", "Custom Job Posting"],
        index=0
    )
    target_role_title = "Senior Software Engineer"
    if job_mode == "Standard Tech Roles":
        selected_job = st.selectbox(
            "Target Position:",
            list(SAMPLE_JOB_DESCRIPTIONS.keys()),
            index=0
        )
        job_description = SAMPLE_JOB_DESCRIPTIONS[selected_job]
        target_role_title = selected_job
    else:
        target_role_title = st.text_input("Target Role Title:", "Staff Machine Learning Engineer")
        job_description = st.text_area(
            "Paste Job Requirements:",
            value=SAMPLE_JOB_DESCRIPTIONS[next(iter(SAMPLE_JOB_DESCRIPTIONS.keys()))],
            height=160
        )

    st.markdown("---")
    st.markdown("#### 🔒 System Capabilities")
    st.caption("• **100% Offline Core ML**\n• **Zero External API Fees**\n• **PDF & DOCX Support**\n• **Recruiter Batch Leaderboard**\n• **BM25 Okapi & PageRank TextRank**")


# ---------------------------------------------------------
# Core Pipeline Execution (Cached / Stateful)
# ---------------------------------------------------------
contact_info = ResumeParser.extract_contact_info(resume_text)
sections = ResumeParser.segment_sections(resume_text)
bullets = ResumeParser.extract_bullet_points(resume_text)
skills = extract_skills_from_text(resume_text)
readability = ResumePreprocessor.calculate_readability(resume_text)
bullet_analysis = ResumePreprocessor.analyze_bullets(bullets)

ats_evaluation = ATSScorer.evaluate(
    contact_info=contact_info,
    sections=sections,
    bullet_analysis=bullet_analysis,
    skills_by_category=skills,
    readability=readability
)

domain_prediction = ml_stack["domain_clf"].predict(resume_text)
fit_analysis = ml_stack["matcher"].evaluate_fit(resume_text, job_description)
seniority_info = SeniorityEstimator.estimate_seniority(
    resume_text=resume_text,
    predicted_domain=domain_prediction["predicted_domain"],
    bullet_analysis=bullet_analysis
)


# ---------------------------------------------------------
# Brand Header & Top KPI Metric Cards
# ---------------------------------------------------------
render_header(title=APP_TITLE, subtitle=APP_SUBTITLE, version=VERSION)
render_kpis(
    ats_eval=ats_evaluation,
    fit_eval=fit_analysis,
    domain_eval=domain_prediction,
    bullet_eval=bullet_analysis
)


# ---------------------------------------------------------
# Tabbed Workspace
# ---------------------------------------------------------
(
    tab_ats,
    tab_match,
    tab_tailor,
    tab_cover,
    tab_seniority,
    tab_batch,
    tab_optimizer,
    tab_qa,
    tab_explorer,
    tab_export
) = st.tabs([
    "📊 ATS Scorecard",
    "🎯 Job Fit & Gaps",
    "✨ AI Resume Tailor",
    "✉️ Cover Letter",
    "📈 Seniority & Roadmap",
    "👥 Batch Leaderboard",
    "✍️ Bullet Optimizer",
    "🤖 Adaptive Q&A",
    "🧠 ML Explorer",
    "📑 Full Report"
])

with tab_ats:
    render_tab_ats(
        ats_evaluation=ats_evaluation,
        contact_info=contact_info,
        readability=readability,
        bullet_analysis=bullet_analysis
    )

with tab_match:
    render_tab_match(fit_analysis=fit_analysis)

with tab_tailor:
    render_tab_tailor(
        resume_text=resume_text,
        fit_analysis=fit_analysis,
        candidate_name=contact_info["name"]
    )

with tab_cover:
    render_tab_cover(
        candidate_name=contact_info["name"],
        contact_info=contact_info,
        target_role_title=target_role_title,
        job_text=job_description,
        bullets=bullets,
        matched_skills=fit_analysis["matched_skills"],
        summarizer=ml_stack["summarizer"]
    )

with tab_seniority:
    render_tab_seniority(
        resume_text=resume_text,
        predicted_domain=domain_prediction["predicted_domain"],
        bullet_analysis=bullet_analysis
    )

with tab_batch:
    render_tab_batch(
        job_description=job_description,
        ml_stack=ml_stack
    )

with tab_optimizer:
    render_tab_optimizer(bullets=bullets)

with tab_qa:
    render_tab_qa(
        resume_text=resume_text,
        job_description=job_description,
        qa_engine=ml_stack["qa_engine"]
    )

with tab_explorer:
    render_tab_ml_explorer(
        resume_text=resume_text,
        bullets=bullets,
        domain_prediction=domain_prediction,
        ml_stack=ml_stack
    )

with tab_export:
    render_tab_export(
        contact_info=contact_info,
        domain_prediction=domain_prediction,
        ats_evaluation=ats_evaluation,
        seniority_info=seniority_info,
        readability=readability,
        bullet_analysis=bullet_analysis,
        fit_analysis=fit_analysis,
        skills=skills,
        job_mode=job_mode
    )
