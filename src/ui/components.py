"""
Modular UI Components for AdaptiveML Enterprise Streamlit Application
"""

import json
from typing import Dict, List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.algorithms.ats_scorer import ATSScorer
from src.algorithms.cover_letter import CoverLetterGenerator
from src.algorithms.matcher import JobFitMatcher
from src.algorithms.seniority_estimator import SeniorityEstimator
from src.algorithms.tailor import ResumeTailor
from src.config import POWER_ACTION_VERBS
from src.data_pipeline.parser import ResumeParser
from src.data_pipeline.preprocessor import ResumePreprocessor
from src.data_pipeline.skill_taxonomy import extract_skills_from_text
from src.sample_data import SAMPLE_RESUMES


def render_header(title: str, subtitle: str, version: str):
    """Render top brand identity and status tags."""
    st.markdown(f"""
    <div class="app-header-container">
        <div style="display:flex; align-items:baseline; justify-content:space-between; flex-wrap:wrap;">
            <div class="brand-title">{title} <span class="version-tag">v{version}</span></div>
            <div style="font-size:0.8rem; color:#94A3B8; font-family:'JetBrains Mono', monospace;">
                🟢 100% Offline Core ML • 0 API Dependencies
            </div>
        </div>
        <div class="brand-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def render_kpis(ats_eval: Dict, fit_eval: Dict, domain_eval: Dict, bullet_eval: Dict):
    """Render high-signal KPI metrics cards."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Overall ATS Score</div>
            <div class="kpi-metric">{ats_eval['total_score']}<span style="font-size:1.1rem; color:#64748B">/100</span></div>
            <div class="kpi-caption text-emerald">● {ats_eval['grade']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Target Job Match</div>
            <div class="kpi-metric">{fit_eval['composite_score']}%</div>
            <div class="kpi-caption text-indigo">● {len(fit_eval['matched_skills'])} of {len(fit_eval['matched_skills']) + len(fit_eval['missing_skills'])} Skills Matched</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Predicted Domain</div>
            <div class="kpi-metric" style="font-size:1.35rem">{domain_eval['predicted_domain']}</div>
            <div class="kpi-caption text-purple">● {domain_eval['confidence']}% ML Confidence</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Quantifiable Impact</div>
            <div class="kpi-metric">{bullet_eval['quantifiable_ratio']}%</div>
            <div class="kpi-caption text-amber">● {bullet_eval['quantifiable_count']} of {bullet_eval['total_bullets']} Bullets with Metrics</div>
        </div>
        """, unsafe_allow_html=True)


def render_tab_ats(ats_evaluation: Dict, contact_info: Dict, readability: Dict, bullet_analysis: Dict):
    """Render ATS Compliance Audit with Radar Chart and Bullet Audit."""
    st.markdown('<div class="view-heading">📊 ATS Multi-Dimensional Compliance Audit</div>', unsafe_allow_html=True)
    st.markdown('<div class="view-sub">Deterministic 7-pillar evaluation matching recruiter applicant tracking benchmarks.</div>', unsafe_allow_html=True)

    col_chart, col_details = st.columns([1.2, 1])

    with col_chart:
        categories = list(ats_evaluation["category_scores"].keys())
        values = list(ats_evaluation["category_scores"].values())
        max_values = [10, 15, 15, 20, 20, 10, 10]
        norm_values = [(v / m) * 100 for v, m in zip(values, max_values)]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=norm_values + [norm_values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(99, 102, 241, 0.22)',
            line=dict(color='#818CF8', width=2),
            name='Score %'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], color="#94A3B8"),
                angularaxis=dict(color="#F8FAFC")
            ),
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=360,
            margin=dict(l=40, r=40, t=25, b=25)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_details:
        st.markdown("#### 🔍 Contact Verification")
        st.write(f"• **Candidate Name:** `{contact_info['name']}`")
        st.write(f"• **Email:** `{contact_info['email'] or 'Missing ⚠️'}`")
        st.write(f"• **Phone:** `{contact_info['phone'] or 'Missing ⚠️'}`")
        st.write(f"• **LinkedIn:** `{contact_info['linkedin'] or 'Missing ⚠️'}`")
        st.write(f"• **GitHub / Portfolio:** `{contact_info['github'] or 'Missing ⚠️'}`")

        st.markdown("#### 📖 Document Readability & Density")
        st.write(f"• **Word Count:** `{readability['total_words']}` words ({readability['total_sentences']} sentences)")
        st.write(f"• **Flesch Reading Ease:** `{readability['flesch_reading_ease']}/100` (Grade Level: `{readability['grade_level']}`)")
        st.write(f"• **Lexical Diversity (TTR):** `{readability['lexical_diversity']}%`")

    st.markdown("---")
    col_str, col_rec = st.columns(2)

    with col_str:
        st.markdown("#### ✅ Validated Strengths")
        for s in ats_evaluation["strengths"]:
            st.success(f"• {s}")

    with col_rec:
        st.markdown("#### 💡 High-Yield Recommendations")
        for r in ats_evaluation["recommendations"]:
            st.warning(f"• {r}")

    st.markdown("---")
    st.markdown("#### 🔬 Experience Bullet Quality Audit")
    if bullet_analysis["bullet_details"]:
        bullet_df = pd.DataFrame([
            {
                "Bullet Point": b["bullet"][:120] + ("..." if len(b["bullet"]) > 120 else ""),
                "Action Verb": "✅ " + b["detected_verb"].title() if b["has_action_verb"] else "❌ Missing",
                "Metric": "✅ Present" if b["has_metric"] else "❌ None",
                "Words": b["word_count"],
                "Quality": f"{b['strength_score']}%"
            }
            for b in bullet_analysis["bullet_details"][:10]
        ])
        st.dataframe(bullet_df, use_container_width=True)


def render_tab_match(fit_analysis: Dict):
    """Render Job Fit Match and Skill Gap Breakdown."""
    st.markdown('<div class="view-heading">🎯 Job Description Fit & Skill Gap Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="view-sub">Synthesizes dense semantic similarity, TF-IDF lexical match, and skill taxonomy overlap.</div>', unsafe_allow_html=True)

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.metric("Composite Fit Score", f"{fit_analysis['composite_score']}%")
    with mcol2:
        st.metric("Semantic Similarity", f"{fit_analysis['semantic_score']}%")
    with mcol3:
        st.metric("Direct Skill Overlap", f"{fit_analysis['skill_match_ratio']}%")
    with mcol4:
        st.metric("Lexical TF-IDF Match", f"{fit_analysis['lexical_score']}%")

    st.markdown("---")
    col_matched, col_missing = st.columns(2)

    with col_matched:
        st.markdown(f"#### ✅ Matched Required Skills ({len(fit_analysis['matched_skills'])})")
        if fit_analysis['matched_skills']:
            pills = "".join([f'<span class="badge-pill badge-matched">{s}</span>' for s in fit_analysis['matched_skills']])
            st.markdown(pills, unsafe_allow_html=True)
        else:
            st.info("No exact skill overlap detected.")

    with col_missing:
        st.markdown(f"#### ⚠️ Missing Skills Required by Role ({len(fit_analysis['missing_skills'])})")
        if fit_analysis['missing_skills']:
            pills = "".join([f'<span class="badge-pill badge-missing">{s}</span>' for s in fit_analysis['missing_skills']])
            st.markdown(pills, unsafe_allow_html=True)
        else:
            st.success("No missing required skills! Candidate covers all primary requirements.")

    st.markdown("---")
    st.markdown("#### 🎯 Priority Keywords to Add to Pass ATS Filters")
    st.caption("Top TF-IDF weighted terms in the Job Description absent from the resume:")
    if fit_analysis["priority_missing_keywords"]:
        kw_html = "".join([f'<span class="badge-pill badge-keyword">{kw}</span>' for kw in fit_analysis["priority_missing_keywords"]])
        st.markdown(kw_html, unsafe_allow_html=True)
    else:
        st.write("• Resume covers primary keyword vocabulary from the posting.")


def render_tab_tailor(resume_text: str, fit_analysis: Dict, candidate_name: str):
    """Render One-Click AI Resume Tailor and DOCX Generator."""
    st.markdown('<div class="view-heading">✨ One-Click AI Resume Tailor & DOCX Exporter</div>', unsafe_allow_html=True)
    st.markdown('<div class="view-sub">Weaves target role competencies and action verbs into your resume to maximize ATS callback rates.</div>', unsafe_allow_html=True)

    tailored_data = ResumeTailor.tailor_resume(resume_text, fit_analysis)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Original ATS Score", f"{tailored_data['ats_before']}/100")
    with c2:
        gain = round(tailored_data['ats_after'] - tailored_data['ats_before'], 1)
        st.metric("Tailored ATS Score", f"{tailored_data['ats_after']}/100", f"+{gain} pts")
    with c3:
        st.metric("Aligned Competencies", f"{len(tailored_data['injected_skills'])} Skills")

    st.markdown("---")
    docx_bytes = ResumeTailor.generate_docx(tailored_data["tailored_text"], candidate_name)

    dcol1, dcol2 = st.columns(2)
    with dcol1:
        st.download_button(
            label="📥 Download Tailored Resume (.DOCX)",
            data=docx_bytes,
            file_name=f"Tailored_Resume_{candidate_name.replace(' ', '_')}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True
        )
    with dcol2:
        st.download_button(
            label="📥 Download Tailored Text (.TXT)",
            data=tailored_data["tailored_text"],
            file_name=f"Tailored_Resume_{candidate_name.replace(' ', '_')}.txt",
            mime="text/plain",
            use_container_width=True
        )

    if tailored_data["changes_log"]:
        with st.expander(f"🔍 Inspect {len(tailored_data['changes_log'])} Automatic Modifications"):
            for ch in tailored_data["changes_log"]:
                st.write(f"• **{ch['type']}**: *'{ch['original']}'* ➔ **'{ch['replacement']}'** ({ch['context']})")

    st.markdown("#### 📄 Tailored Resume Preview")
    st.text_area("Output Preview:", value=tailored_data["tailored_text"], height=320)


def render_tab_cover(candidate_name: str, contact_info: Dict, target_role_title: str, job_text: str, bullets: List[str], matched_skills: List[str], summarizer):
    """Render Algorithmic Cover Letter Generator."""
    st.markdown('<div class="view-heading">✉️ Algorithmic Tailored Cover Letter Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="view-sub">Synthesizes a structured 3-part cover letter connecting candidate achievements to company requirements.</div>', unsafe_allow_html=True)

    top_acc = [b[2] for b in summarizer.summarize(bullets, top_n=2)]
    letter_text = CoverLetterGenerator.generate_cover_letter(
        candidate_name=candidate_name,
        contact_info=contact_info,
        target_role=target_role_title,
        job_text=job_text,
        top_accomplishments=top_acc,
        matched_skills=matched_skills
    )

    letter_docx = CoverLetterGenerator.generate_docx(letter_text, candidate_name)

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            label="📥 Download Cover Letter (.DOCX)",
            data=letter_docx,
            file_name=f"Cover_Letter_{candidate_name.replace(' ', '_')}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True
        )
    with d2:
        st.download_button(
            label="📥 Download Cover Letter (.TXT)",
            data=letter_text,
            file_name=f"Cover_Letter_{candidate_name.replace(' ', '_')}.txt",
            mime="text/plain",
            use_container_width=True
        )

    st.markdown("#### 📝 Editable Draft")
    st.text_area("Cover Letter Editor:", value=letter_text, height=360)


def render_tab_seniority(resume_text: str, predicted_domain: str, bullet_analysis: Dict):
    """Render Seniority Estimator and Career Progression Roadmap."""
    st.markdown('<div class="view-heading">📈 ML Seniority Estimator & Career Progression Roadmap</div>', unsafe_allow_html=True)
    st.markdown('<div class="view-sub">Predicts candidate engineering tier, benchmarks market base compensation, and reveals milestones for next-level promotions.</div>', unsafe_allow_html=True)

    seniority_info = SeniorityEstimator.estimate_seniority(
        resume_text=resume_text,
        predicted_domain=predicted_domain,
        bullet_analysis=bullet_analysis
    )

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Estimated Seniority Band</div>
            <div class="kpi-metric" style="font-size:1.6rem">{seniority_info['seniority_level']}</div>
            <div class="kpi-caption text-indigo">~{seniority_info['estimated_years_experience']} Years Career Span</div>
        </div>
        """, unsafe_allow_html=True)

    with s2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Market Base Salary Band</div>
            <div class="kpi-metric" style="font-size:1.35rem">{seniority_info['salary_benchmark']}</div>
            <div class="kpi-caption text-emerald">US Benchmark ({predicted_domain})</div>
        </div>
        """, unsafe_allow_html=True)

    with s3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Target Promotion Tier</div>
            <div class="kpi-metric" style="font-size:1.35rem">{seniority_info['next_level']}</div>
            <div class="kpi-caption text-purple">{seniority_info['leadership_bullet_count']} Leadership Bullets Identified</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    col_tech, col_lead = st.columns(2)

    with col_tech:
        st.markdown("#### 🛠️ Technical Competency Milestones to Level Up")
        for ms in seniority_info["technical_milestones"]:
            st.info(f"• {ms}")

    with col_lead:
        st.markdown("#### 👥 Leadership & Architecture Milestones")
        for ms in seniority_info["leadership_milestones"]:
            st.success(f"• {ms}")


def render_tab_batch(job_description: str, ml_stack: Dict):
    """Render Recruiter Multi-Resume Batch Leaderboard."""
    st.markdown('<div class="view-heading">👥 Recruiter Multi-Resume Batch Leaderboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="view-sub">Evaluate and rank multiple candidate resumes simultaneously against target role criteria.</div>', unsafe_allow_html=True)

    source_choice = st.radio(
        "Candidate Batch Source:",
        ["⚡ Preloaded 5 Diverse Candidates (Alex, Sarah, Marcus, Priya, Jordan)", "📤 Upload Multiple Resumes (PDF, DOCX, TXT)"],
        horizontal=True
    )

    batch_resumes = {}
    if source_choice.startswith("⚡"):
        batch_resumes = SAMPLE_RESUMES
    else:
        uploaded_multi = st.file_uploader(
            "Upload 2 to 10 Resumes at once:",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )
        if uploaded_multi:
            for f in uploaded_multi:
                batch_resumes[f.name] = ResumeParser.parse_file(f.name, f.read())
        else:
            st.info("Upload multiple resumes above or explore with preloaded candidates.")
            batch_resumes = SAMPLE_RESUMES

    if batch_resumes:
        leaderboard_data = []
        with st.spinner("Processing batch candidates through ML pipelines..."):
            for cand_name, cand_text in batch_resumes.items():
                c_contact = ResumeParser.extract_contact_info(cand_text)
                c_sections = ResumeParser.segment_sections(cand_text)
                c_bullets = ResumeParser.extract_bullet_points(cand_text)
                c_skills = extract_skills_from_text(cand_text)
                c_readability = ResumePreprocessor.calculate_readability(cand_text)
                c_bullet_analysis = ResumePreprocessor.analyze_bullets(c_bullets)

                c_ats = ATSScorer.evaluate(c_contact, c_sections, c_bullet_analysis, c_skills, c_readability)
                c_fit = ml_stack["matcher"].evaluate_fit(cand_text, job_description)
                c_domain = ml_stack["domain_clf"].predict(cand_text)["predicted_domain"]

                composite = c_fit["composite_score"]
                if composite >= 75.0:
                    status = "🟢 Tier 1 (Strong Hire)"
                elif composite >= 60.0:
                    status = "🟡 Tier 2 (Interview)"
                elif composite >= 45.0:
                    status = "🟠 Tier 3 (Review)"
                else:
                    status = "🔴 Screen Out"

                leaderboard_data.append({
                    "Candidate": c_contact["name"] if c_contact["name"] != "Candidate" else cand_name.split("(")[0].strip(),
                    "Domain": c_domain,
                    "Composite Fit %": composite,
                    "ATS Score": c_ats["total_score"],
                    "Matched Skills": len(c_fit["matched_skills"]),
                    "Missing Skills": len(c_fit["missing_skills"]),
                    "Top Skills Matched": ", ".join(c_fit["matched_skills"][:4]) or "None",
                    "Recommendation": status
                })

        leaderboard_df = pd.DataFrame(leaderboard_data).sort_values(by="Composite Fit %", ascending=False).reset_index(drop=True)
        leaderboard_df.index = leaderboard_df.index + 1
        leaderboard_df.index.name = "Rank"

        st.markdown("### 🏆 Candidate Rankings")
        st.dataframe(
            leaderboard_df[["Candidate", "Domain", "Composite Fit %", "ATS Score", "Matched Skills", "Missing Skills", "Recommendation"]],
            use_container_width=True
        )

        fig_scatter = px.scatter(
            leaderboard_df,
            x="ATS Score",
            y="Composite Fit %",
            text="Candidate",
            color="Recommendation",
            size="Matched Skills",
            size_max=30,
            color_discrete_map={
                "🟢 Tier 1 (Strong Hire)": "#10B981",
                "🟡 Tier 2 (Interview)": "#F59E0B",
                "🟠 Tier 3 (Review)": "#F97316",
                "🔴 Screen Out": "#EF4444"
            },
            title="Candidate Decision Matrix: Job Fit % vs ATS Score"
        )
        fig_scatter.update_traces(textposition="top center", marker=dict(line=dict(width=1, color="#FFFFFF")))
        fig_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(range=[20, 105], title="ATS Score (0-100)"),
            yaxis=dict(range=[20, 105], title="Job Description Fit %"),
            height=400
        )
        st.plotly_chart(fig_scatter, use_container_width=True)


def render_tab_optimizer(bullets: List[str]):
    """Render Real-Time Live Bullet Point Optimizer."""
    st.markdown('<div class="view-heading">✍️ Interactive Live Bullet Point Optimizer</div>', unsafe_allow_html=True)
    st.markdown('<div class="view-sub">Draft or refine your experience bullet points in real-time. Watch your strength score, action verbs, and quantifiable metrics update instantly.</div>', unsafe_allow_html=True)

    sample_bullet = (
        bullets[0] if bullets else "Optimized transformer inference latency by 42% utilizing TensorRT and Triton, saving $140,000 annually."
    )

    col_edit, col_feedback = st.columns([1.3, 1])

    with col_edit:
        bullet_input = st.text_area(
            "Draft / Edit Your Resume Bullet Point:",
            value=sample_bullet,
            height=125,
            help="Optimal format: Action Verb + Project Scope + Measurable Result."
        )

        st.markdown("#### 🚀 Power Action Verb Library")
        cat_choice = st.selectbox("Select Verb Category:", list(POWER_ACTION_VERBS.keys()))
        pills = "".join([f'<span class="badge-pill badge-keyword">{v.title()}</span>' for v in POWER_ACTION_VERBS[cat_choice]])
        st.markdown(pills, unsafe_allow_html=True)

    with col_feedback:
        st.markdown("#### ⚡ Real-Time Diagnostic Feedback")
        single_analysis = ResumePreprocessor.analyze_bullets([bullet_input])
        details = single_analysis["bullet_details"][0] if single_analysis["bullet_details"] else {}

        strength = details.get("strength_score", 0)
        has_verb = details.get("has_action_verb", False)
        verb_str = details.get("detected_verb", None)
        has_metric = details.get("has_metric", False)
        w_count = details.get("word_count", 0)

        st.metric("Bullet Strength Score", f"{strength}/100")
        
        if has_verb:
            st.success(f"✅ Power Action Verb: **{verb_str.title()}**")
        else:
            st.error("❌ Missing Action Verb (Start with an active verb)")

        if has_metric:
            st.success("✅ Quantifiable Metric Detected (Numbers, %, or $)")
        else:
            st.warning("⚠️ No metrics found (Add % improvement, scale, or dollar impact)")

        if 10 <= w_count <= 25:
            st.success(f"✅ Optimal Length ({w_count} words)")
        elif w_count < 10:
            st.warning(f"⚠️ Too brief ({w_count} words). Aim for 12-20 words.")
        else:
            st.warning(f"⚠️ Run-on sentence ({w_count} words).")

        st.markdown("---")
        st.markdown("#### 💡 The Google 'X-Y-Z' Formula")
        st.info("**Template**: *Accomplished [X] as measured by [Y], by doing [Z]*\n\n**Example**: *Reduced API latency by 45% (from 220ms to 120ms) by implementing Redis caching and indexing PostgreSQL tables.*")


def render_tab_qa(resume_text: str, job_description: str, qa_engine):
    """Render Conversational Chat Assistant with Session State Memory."""
    st.markdown('<div class="view-heading">🤖 Adaptive ML Question-Answering Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="view-sub">Replaces traditional RAG with an intelligent ML Intent Classifier, BM25 Okapi retriever, and diagnostic synthesis engine.</div>', unsafe_allow_html=True)

    # Initialize chat history in session state
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": "Hello! I am your Adaptive ML Career Assistant. You can ask me any question about this candidate's background, ATS compliance, role fit, or generate tailored interview questions.",
                "intent": "GREETING",
                "confidence": 100.0,
                "snippets": []
            }
        ]

    # Quick prompt buttons
    st.markdown("**Quick Inquiries:**")
    qc1, qc2, qc3, qc4, qc5 = st.columns(5)
    preset_query = None

    if qc1.button("📊 What is my ATS score?", use_container_width=True):
        preset_query = "What is my ATS score and how can I improve it?"
    if qc2.button("🛠️ List technical skills", use_container_width=True):
        preset_query = "What technical skills, programming languages, and databases does the candidate know?"
    if qc3.button("🎯 Compare with job role", use_container_width=True):
        preset_query = "Does this candidate match the job description and what skills are missing?"
    if qc4.button("🎙️ Generate interview questions", use_container_width=True):
        preset_query = "Generate technical and behavioral interview questions based on their projects"
    if qc5.button("📋 Executive Summary", use_container_width=True):
        preset_query = "Give me an executive summary of this candidate's career trajectory"

    # Display conversation history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            if msg.get("intent") and msg["intent"] != "GREETING":
                st.markdown(f'<div class="intent-indicator">⚡ INTENT: {msg["intent"]} ({msg.get("confidence", 90)}% CONFIDENCE)</div>', unsafe_allow_html=True)
            st.markdown(msg["content"])
            if msg.get("snippets"):
                with st.expander("🔍 Retrieved Evidence Spans (BM25 / Centrality)"):
                    for snip in msg["snippets"]:
                        st.write(f"• *\"{snip}\"*")

    # Chat Input Box
    user_query = st.chat_input("Ask a question about the candidate or target role...")
    query_to_run = preset_query or user_query

    if query_to_run:
        # Append User Message
        st.session_state.chat_messages.append({"role": "user", "content": query_to_run})
        with st.chat_message("user"):
            st.markdown(query_to_run)

        # Generate Assistant Response via ML Pipeline
        with st.spinner("Processing via Adaptive ML Pipeline..."):
            res = qa_engine.answer_query(
                query=query_to_run,
                resume_text=resume_text,
                job_description=job_description
            )

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": res["answer"],
            "intent": res["intent"],
            "confidence": res["confidence"],
            "snippets": res.get("evidence_snippets", [])
        })

        with st.chat_message("assistant"):
            st.markdown(f'<div class="intent-indicator">⚡ INTENT: {res["intent"]} ({res["confidence"]}% CONFIDENCE)</div>', unsafe_allow_html=True)
            st.markdown(res["answer"])
            if res.get("evidence_snippets"):
                with st.expander("🔍 Retrieved Evidence Spans (BM25 / Centrality)"):
                    for snip in res["evidence_snippets"]:
                        st.write(f"• *\"{snip}\"*")


def render_tab_ml_explorer(resume_text: str, bullets: List[str], domain_prediction: Dict, ml_stack: Dict):
    """Render Mathematical Diagnostics and Feature Visualizations."""
    st.markdown('<div class="view-heading">🧠 Machine Learning Pipelines & Mathematical Diagnostics</div>', unsafe_allow_html=True)
    st.markdown('<div class="view-sub">Inspect the algorithmic representations, feature distributions, and supervised/unsupervised models.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 1. Supervised Domain Classifier Probabilities")
        dom_df = pd.DataFrame(
            list(domain_prediction["distribution"].items()),
            columns=["Engineering Domain", "Probability (%)"]
        )
        fig_bar = px.bar(
            dom_df,
            x="Probability (%)",
            y="Engineering Domain",
            orientation="h",
            color="Probability (%)",
            color_continuous_scale="Purples"
        )
        fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(l=20, r=20, t=10, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        st.markdown("#### 2. TF-IDF Top Term Importance")
        top_terms = ml_stack["tfidf_engine"].get_top_terms(resume_text, top_n=8)
        if top_terms:
            tfidf_df = pd.DataFrame(top_terms, columns=["Term", "TF-IDF Weight"])
            fig_tfidf = px.bar(
                tfidf_df,
                x="TF-IDF Weight",
                y="Term",
                orientation="h",
                color="TF-IDF Weight",
                color_continuous_scale="Teal"
            )
            fig_tfidf.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(l=20, r=20, t=10, b=10))
            st.plotly_chart(fig_tfidf, use_container_width=True)

    st.markdown("---")
    tc1, tc2 = st.columns(2)
    with tc1:
        st.markdown("#### 3. Latent Dirichlet Allocation (LDA) Topic Modeler")
        topics = ml_stack["topic_modeler"].fit_transform(bullets)
        if topics:
            for t in topics:
                words_str = ", ".join([f"{w['word']} ({w['weight']})" for w in t["top_words"][:4]])
                st.write(f"**{t['label']}** — *{t['prevalence']}% document weight*")
                st.caption(f"Top terms: {words_str}")
        else:
            st.info("Insufficient bullet points for LDA topic clustering.")

    with tc2:
        st.markdown("#### 4. K-Means Semantic Experience Clusters")
        clusters = ml_stack["cluster_engine"].cluster_bullets(bullets, max_clusters=3)
        for c in clusters:
            with st.expander(f"📁 {c['label']} ({c['size']} bullets)"):
                for b in c["bullets"]:
                    st.write(f"• {b}")


def render_tab_export(contact_info: Dict, domain_prediction: Dict, ats_evaluation: Dict, seniority_info: Dict, readability: Dict, bullet_analysis: Dict, fit_analysis: Dict, skills: Dict, job_mode: str):
    """Render Exportable Comprehensive Markdown and JSON Audit."""
    st.markdown('<div class="view-heading">📑 Full Candidate Audit & Diagnostic Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="view-sub">Export exhaustive audit logs for recruitment files, candidate reviews, and ATS compliance documentation.</div>', unsafe_allow_html=True)

    report_md = f"""# AdaptiveML Candidate Audit Report
**Candidate**: {contact_info['name']}
**Domain Specialization**: {domain_prediction['predicted_domain']} ({domain_prediction['confidence']}% Confidence)
**ATS Score**: {ats_evaluation['total_score']}/100 ({ats_evaluation['grade']})
**Seniority Tier**: {seniority_info['seniority_level']} (Est. {seniority_info['estimated_years_experience']} YOE)
**Market Compensation Band**: {seniority_info['salary_benchmark']}

---

## 1. Contact Verification
- Email: {contact_info['email']}
- Phone: {contact_info['phone']}
- LinkedIn: {contact_info['linkedin']}
- GitHub: {contact_info['github']}

## 2. ATS Component Breakdown
{json.dumps(ats_evaluation['category_scores'], indent=2)}

## 3. Readability & Impact
- Total Words: {readability['total_words']}
- Flesch Reading Ease: {readability['flesch_reading_ease']}
- Action Verb Ratio: {bullet_analysis['action_verb_ratio']}%
- Quantifiable Metrics Ratio: {bullet_analysis['quantifiable_ratio']}%

## 4. Key Strengths
{chr(10).join(['- ' + s for s in ats_evaluation['strengths']])}

## 5. Recommended Actions
{chr(10).join(['- ' + r for r in ats_evaluation['recommendations']])}

## 6. Job Fit Comparison (vs {job_mode})
- Composite Score: {fit_analysis['composite_score']}%
- Matched Skills: {', '.join(fit_analysis['matched_skills'])}
- Missing Skills: {', '.join(fit_analysis['missing_skills'])}
- Priority Keywords to Add: {', '.join(fit_analysis['priority_missing_keywords'])}
"""

    st.download_button(
        label="📥 Download Full Audit Report (.MD)",
        data=report_md,
        file_name=f"ATS_Audit_{contact_info['name'].replace(' ', '_')}.md",
        mime="text/markdown"
    )

    with st.expander("📄 View Parsed Structured JSON Data"):
        st.json({
            "contact": contact_info,
            "readability": readability,
            "domain_prediction": domain_prediction,
            "seniority_estimation": seniority_info,
            "ats_evaluation": ats_evaluation,
            "job_fit": fit_analysis,
            "skills": skills
        })
