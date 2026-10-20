"""
Adaptive ML Question-Answering and Autonomous Diagnostic Engine
Routes user questions using the ML Intent Classifier and generates factual, algorithm-backed answers.
"""

from typing import Any

from src.algorithms.ats_scorer import ATSScorer
from src.algorithms.matcher import JobFitMatcher
from src.algorithms.text_rank import TextRankSummarizer
from src.data_pipeline.parser import ResumeParser
from src.data_pipeline.preprocessor import ResumePreprocessor
from src.data_pipeline.skill_taxonomy import extract_skills_from_text
from src.feature_engineering.embeddings import DenseEmbedder
from src.feature_engineering.vectorizers import BM25Okapi
from src.models.domain_classifier import DomainClassifier
from src.models.intent_classifier import AdaptiveIntentClassifier


class AdaptiveQAEngine:
    """End-to-end adaptive machine learning assistant for resume question-answering and analysis."""

    def __init__(
        self,
        intent_classifier: AdaptiveIntentClassifier,
        domain_classifier: DomainClassifier,
        embedder: DenseEmbedder,
        matcher: JobFitMatcher,
        summarizer: TextRankSummarizer
    ):
        self.intent_classifier = intent_classifier
        self.domain_classifier = domain_classifier
        self.embedder = embedder
        self.matcher = matcher
        self.summarizer = summarizer

    def answer_query(
        self,
        query: str,
        resume_text: str,
        job_description: str | None = None
    ) -> dict[str, Any]:
        """
        Process user query adaptively using machine learning pipelines without requiring paid LLM APIs.
        """
        if not resume_text.strip():
            return {
                "intent": "UNKNOWN",
                "confidence": 0.0,
                "answer": "Please upload a resume or select a sample resume first to begin querying.",
                "evidence_snippets": []
            }

        # 1. Classify Query Intent via Trained ML Model
        intent_data = self.intent_classifier.classify_intent(query)
        intent = intent_data["intent"]
        confidence = intent_data["confidence"]

        # 2. Extract resume structures
        contact_info = ResumeParser.extract_contact_info(resume_text)
        sections = ResumeParser.segment_sections(resume_text)
        bullets = ResumeParser.extract_bullet_points(resume_text)
        skills = extract_skills_from_text(resume_text)
        readability = ResumePreprocessor.calculate_readability(resume_text)
        bullet_analysis = ResumePreprocessor.analyze_bullets(bullets)

        evidence_snippets: list[str] = []
        answer_text = ""

        # Route to appropriate ML processing pipeline
        if intent == "ATS_DIAGNOSTICS":
            ats_result = ATSScorer.evaluate(
                contact_info=contact_info,
                sections=sections,
                bullet_analysis=bullet_analysis,
                skills_by_category=skills,
                readability=readability
            )
            score = ats_result["total_score"]
            grade = ats_result["grade"]

            top_rec = "\n".join([f"• {rec}" for rec in ats_result["recommendations"][:4]])
            top_str = "\n".join([f"• {s}" for s in ats_result["strengths"][:3]])
            top_weak = "\n".join([f"• {w}" for w in ats_result["weaknesses"][:3]])

            answer_text = (
                f"### 📊 ATS Audit & Diagnostic Results\n\n"
                f"**Overall ATS Score**: **{score}/100** — *{grade}*\n\n"
                f"#### 🔍 Key Weaknesses:\n{top_weak if top_weak else '• No critical red flags detected.'}\n\n"
                f"#### 💡 Top Actionable Improvements:\n{top_rec if top_rec else '• Document meets high ATS compliance standards.'}\n\n"
                f"#### ✅ Strong Areas:\n{top_str}"
            )
            evidence_snippets = [f"ATS Score: {score}/100", f"Action Verb Ratio: {bullet_analysis['action_verb_ratio']}%", f"Quantifiable Ratio: {bullet_analysis['quantifiable_ratio']}%"]

        elif intent == "JOB_FIT_MATCH":
            if not job_description or not job_description.strip():
                answer_text = (
                    "### 🎯 Job Description Required\n\n"
                    "To evaluate job fit, please provide a target Job Description in the **Job Description Fit** tab "
                    "or paste the job requirements here. I will then compute semantic fit, hard skill overlap, and missing keywords."
                )
            else:
                fit_result = self.matcher.evaluate_fit(resume_text, job_description)
                comp = fit_result["composite_score"]
                sem = fit_result["semantic_score"]
                matched_s = ", ".join(fit_result["matched_skills"][:6]) or "None"
                missing_s = ", ".join(fit_result["missing_skills"][:6]) or "None"
                keywords = ", ".join(fit_result["priority_missing_keywords"][:5]) or "None"

                answer_text = (
                    f"### 🎯 Role Match & Gap Analysis\n\n"
                    f"**Composite Match Score**: **{comp}%**\n"
                    f"- **Semantic Concept Similarity**: {sem}%\n"
                    f"- **Direct Skill Overlap**: {fit_result['skill_match_ratio']}%\n\n"
                    f"**Matched Required Skills**: `{matched_s}`\n\n"
                    f"**Missing Required Skills**: `{missing_s}`\n\n"
                    f"**Top Missing Keywords to Add**: `{keywords}`"
                )
                evidence_snippets = [f"Matched Skills: {matched_s}", f"Missing Skills: {missing_s}"]

        elif intent == "INTERVIEW_PREP":
            # Generate algorithmic interview questions based on candidate's skills and top bullets
            domain_pred = self.domain_classifier.predict(resume_text)["predicted_domain"]
            sample_skills = [s for sublist in skills.values() for s in sublist][:5]
            skills_str = ", ".join(sample_skills) if sample_skills else "software engineering"

            # Pick top 2 high-impact bullets
            ranked_bullets = self.summarizer.summarize(bullets, top_n=2)
            bullet_prompts = [b[2] for b in ranked_bullets]

            q1 = f"**Technical Deep-Dive**: You listed expertise in **{skills_str}**. Can you explain how you made architectural decisions or optimized bottlenecks using these technologies?"
            q2 = f"**Project Execution**: Regarding your accomplishment: *\"{bullet_prompts[0] if bullet_prompts else 'your primary project'}\"* — What technical challenges arose, and how did you measure your results?"
            q3 = f"**System Design & Reliability**: For a **{domain_pred}** role, how would you design a scalable, fault-tolerant system with automated CI/CD and observability?"
            q4 = "**Behavioral / STAR Framework**: Tell me about a time you had to balance technical debt against urgent feature delivery. How did you align with stakeholders?"

            answer_text = (
                f"### 🎙️ Tailored Interview Preparation\n"
                f"*Target Domain: **{domain_pred}***\n\n"
                f"1. {q1}\n\n"
                f"2. {q2}\n\n"
                f"3. {q3}\n\n"
                f"4. {q4}"
            )
            evidence_snippets = bullet_prompts

        elif intent == "EXECUTIVE_SUMMARY":
            domain_info = self.domain_classifier.predict(resume_text)
            pred_domain = domain_info["predicted_domain"]
            confidence_dom = domain_info["confidence"]

            ranked = self.summarizer.summarize(bullets, top_n=3)
            bullet_points_summary = "\n".join([f"• {b[2]}" for b in ranked])

            answer_text = (
                f"### 📋 Candidate Executive Profile\n\n"
                f"- **Primary Domain Specialization**: **{pred_domain}** ({confidence_dom}% ML confidence)\n"
                f"- **Verified Technical Skills**: {sum(len(v) for v in skills.values())} tools across {len(skills)} tech categories\n"
                f"- **Quantifiable Impact Ratio**: {bullet_analysis['quantifiable_ratio']}% of experience bullets\n\n"
                f"#### 🏆 Top Accomplishments (Extracted via TextRank Centrality):\n"
                f"{bullet_points_summary}"
            )
            evidence_snippets = [b[2] for b in ranked]

        else: # SPECIFIC_DETAIL or Fallback
            # Use BM25 + Cosine similarity over all resume sentences/bullets
            sentences = [s.strip() for s in resume_text.split("\n") if len(s.strip().split()) >= 3]
            if not sentences:
                sentences = bullets

            bm25 = BM25Okapi(sentences)
            ranked_results = bm25.rank(query, top_k=4)

            if ranked_results:
                matched_items = []
                for _, score, text in ranked_results:
                    matched_items.append(f"• **{text}** *(BM25 score: {round(score, 2)})*")
                    evidence_snippets.append(text)

                snippets_formatted = "\n".join(matched_items)
                answer_text = (
                    f"### 🔍 Extracted Facts & Context\n\n"
                    f"Based on exact semantic indexing of the uploaded resume, here are the most relevant sections:\n\n"
                    f"{snippets_formatted}"
                )
            else:
                answer_text = (
                    f"### 🔍 Information Retrieval\n\n"
                    f"I scanned the resume for terms related to *'{query}'*, but found no direct references. "
                    f"You may check the detected skills matrix or contact details in the overview tabs."
                )

        return {
            "intent": intent,
            "confidence": confidence,
            "answer": answer_text,
            "evidence_snippets": evidence_snippets
        }
