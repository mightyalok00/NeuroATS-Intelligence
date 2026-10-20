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
from src.data_pipeline.skill_taxonomy import SKILL_TAXONOMY, extract_skills_from_text
from src.feature_engineering.embeddings import DenseEmbedder
from src.feature_engineering.vectorizers import TFIDFEngine
from src.models.cluster_engine import ExperienceClusterEngine
from src.models.domain_classifier import DomainClassifier
from src.models.intent_classifier import AdaptiveIntentClassifier
from src.models.topic_modeler import TopicModeler
from src.sample_data import SAMPLE_JOB_DESCRIPTIONS, SAMPLE_RESUMES
from src.ui import (
    ENTERPRISE_CSS,
    render_active_resume_banner,
    render_filter_banner,
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
# State Initialization
# ---------------------------------------------------------
if "resume_text" not in st.session_state:
    default_sample_key = next(iter(SAMPLE_RESUMES.keys()))
    st.session_state["resume_text"] = SAMPLE_RESUMES[default_sample_key]
    st.session_state["candidate_file_name"] = default_sample_key
    st.session_state["source_mode"] = "Sample Profile"
    st.session_state["is_uploaded"] = False

# ---------------------------------------------------------
# Sidebar Controls & Document Ingestion
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 📂 Input Selection")
    
    input_mode = st.radio(
        "Source Method:",
        ["⚡ Preloaded High-Quality Sample", "📤 Upload Custom PDF / DOCX / TXT", "✍️ Paste Resume Text"],
        index=1 if st.session_state.get("is_uploaded", False) else 0,
        key="sidebar_source_method"
    )

    if input_mode == "⚡ Preloaded High-Quality Sample":
        chosen_sample = st.selectbox(
            "Select Candidate Profile:",
            list(SAMPLE_RESUMES.keys()),
            index=0,
            key="sidebar_sample_select"
        )
        btn_load_sample = st.button("🔄 Load Candidate Profile", use_container_width=True, key="btn_load_sample")
        if btn_load_sample or (not st.session_state.get("is_uploaded", False) and st.session_state.get("last_chosen_sample") != chosen_sample):
            st.session_state["resume_text"] = SAMPLE_RESUMES[chosen_sample]
            st.session_state["candidate_file_name"] = chosen_sample
            st.session_state["source_mode"] = "Sample Profile"
            st.session_state["is_uploaded"] = False
            st.session_state["last_chosen_sample"] = chosen_sample
            if btn_load_sample:
                st.rerun()

    elif input_mode == "📤 Upload Custom PDF / DOCX / TXT":
        uploaded_file = st.file_uploader(
            "Upload Resume (PDF, DOCX, TXT):",
            type=["pdf", "docx", "txt"],
            key="sidebar_uploader_widget",
            help="Your file is parsed 100% locally on your machine with zero cloud upload."
        )
        btn_analyze_sidebar = st.button("⚡ Analyse Uploaded Resume", type="primary", use_container_width=True, key="btn_analyze_sidebar")
        if uploaded_file is not None and (
            btn_analyze_sidebar or st.session_state.get("last_sidebar_upload") != uploaded_file.name
        ):
            extracted = ResumeParser.parse_file(uploaded_file.name, uploaded_file.read())
            if extracted and not extracted.startswith("Error"):
                st.session_state["resume_text"] = extracted
                st.session_state["candidate_file_name"] = uploaded_file.name
                st.session_state["source_mode"] = "Uploaded File"
                st.session_state["is_uploaded"] = True
                st.session_state["last_sidebar_upload"] = uploaded_file.name
                st.rerun()
            else:
                st.error(extracted)

    else:
        pasted = st.text_area("Paste Resume Text Here:", height=240, key="sidebar_paste_area")
        btn_analyze_paste = st.button("⚡ Analyse Pasted Resume", type="primary", use_container_width=True, key="btn_analyze_paste")
        if btn_analyze_paste and pasted.strip():
            st.session_state["resume_text"] = pasted
            st.session_state["candidate_file_name"] = "Pasted Resume Text"
            st.session_state["source_mode"] = "Pasted Text"
            st.session_state["is_uploaded"] = False
            st.rerun()

    resume_text = st.session_state["resume_text"]
    candidate_file_name = st.session_state["candidate_file_name"]

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
    st.markdown("### 🎛️ Pipeline Filters & Cutoffs")
    with st.expander("Scoring & Taxonomy Filters", expanded=True):
        min_ats_threshold = st.slider(
            "Target Passing ATS Cutoff:",
            min_value=50,
            max_value=95,
            value=75,
            step=5,
            help="Sets the target threshold for compliance pass/fail indicator."
        )
        domain_filter = st.selectbox(
            "Target Domain Focus:",
            ["All Tech Domains", "Machine Learning & AI", "Full-Stack & Backend", "DevOps & Cloud Infrastructure", "Data Engineering & Analytics"],
            index=0,
            help="Benchmarks candidate alignment against the selected engineering discipline."
        )
        all_taxonomies = list(SKILL_TAXONOMY.keys())
        selected_taxonomies = st.multiselect(
            "Filter Skill Taxonomy Families:",
            options=all_taxonomies,
            default=all_taxonomies,
            help="Isolate technical skills by domain/family."
        )
        keyword_search = st.text_input(
            "Quick Keyword Search Filter:",
            placeholder="e.g. PyTorch, Docker, React",
            help="Highlight and filter terms across candidate profile."
        )

    st.markdown("---")
    st.markdown("#### 🔒 System Capabilities")
    st.caption("• **100% Offline Core ML**\n• **Zero External API Fees**\n• **PDF & DOCX Support**\n• **Recruiter Batch Leaderboard**\n• **BM25 Okapi & PageRank TextRank**")
    st.markdown("🌐 **Live App:** [neuroats-intelligence.streamlit.app](https://neuroats-intelligence.streamlit.app/)")
    st.markdown("⭐ **GitHub:** [mightyalok00/NeuroATS-Intelligence](https://github.com/mightyalok00/NeuroATS-Intelligence)")


# ---------------------------------------------------------
# Core Pipeline Execution (Cached / Stateful)
# ---------------------------------------------------------
contact_info = ResumeParser.extract_contact_info(resume_text)
sections = ResumeParser.segment_sections(resume_text)
bullets = ResumeParser.extract_bullet_points(resume_text)
skills = extract_skills_from_text(resume_text)
readability = ResumePreprocessor.calculate_readability(resume_text)
bullet_analysis = ResumePreprocessor.analyze_bullets(bullets)

filtered_skills = {
    cat: s_list for cat, s_list in skills.items()
    if cat in selected_taxonomies
}
total_skills_count = sum(len(v) for v in skills.values())
filtered_skills_count = sum(len(v) for v in filtered_skills.values())

ats_evaluation = ATSScorer.evaluate(
    contact_info=contact_info,
    sections=sections,
    bullet_analysis=bullet_analysis,
    skills_by_category=filtered_skills if filtered_skills else skills,
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
# Brand Header, Active Profile & Upload Panel
# ---------------------------------------------------------
render_header(title=APP_TITLE, subtitle=APP_SUBTITLE, version=VERSION)

render_active_resume_banner(
    candidate_name=st.session_state["candidate_file_name"],
    source_label=st.session_state["source_mode"],
    word_count=len(resume_text.split()),
    is_uploaded=st.session_state["is_uploaded"]
)

with st.expander("📤 Upload & Analyse Your Resume (PDF, DOCX, TXT)", expanded=not st.session_state["is_uploaded"]):
    u_col1, u_col2 = st.columns([3, 1.2])
    with u_col1:
        main_upload = st.file_uploader(
            "Select Resume File (PDF, DOCX, TXT):",
            type=["pdf", "docx", "txt"],
            key="main_resume_uploader_file",
            help="100% offline analysis. Your document is processed strictly in local memory with zero external cloud calls."
        )
    with u_col2:
        st.write("")
        st.write("")
        btn_main_analyze = st.button("🚀 Analyse My Resume", type="primary", use_container_width=True, key="btn_main_analyze")

    if main_upload is not None and (
        btn_main_analyze or st.session_state.get("last_uploaded_main") != main_upload.name
    ):
        extracted_text = ResumeParser.parse_file(main_upload.name, main_upload.read())
        if extracted_text and not extracted_text.startswith("Error"):
            st.session_state["resume_text"] = extracted_text
            st.session_state["candidate_file_name"] = main_upload.name
            st.session_state["source_mode"] = "Uploaded File"
            st.session_state["is_uploaded"] = True
            st.session_state["last_uploaded_main"] = main_upload.name
            st.success(f"✅ Successfully loaded '{main_upload.name}'! Running full offline AI analysis...")
            st.rerun()
        else:
            st.error(extracted_text)

render_kpis(
    ats_eval=ats_evaluation,
    fit_eval=fit_analysis,
    domain_eval=domain_prediction,
    bullet_eval=bullet_analysis
)
render_filter_banner(
    ats_score=ats_evaluation["total_score"],
    threshold=min_ats_threshold,
    domain_pred=domain_prediction["predicted_domain"],
    target_domain=domain_filter,
    total_skills_count=total_skills_count,
    filtered_skills_count=filtered_skills_count,
    keyword_filter=keyword_search.strip()
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
