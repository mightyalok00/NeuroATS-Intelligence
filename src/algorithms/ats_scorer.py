"""
Multi-Dimensional ATS Scoring Engine
Evaluates structural integrity, quantifiable impact, action verbs, and skill depth.
"""

from typing import Any


class ATSScorer:
    """Computes a multi-dimensional ATS score out of 100 with itemized audit points."""

    @classmethod
    def evaluate(
        cls,
        contact_info: dict[str, Any],
        sections: dict[str, str],
        bullet_analysis: dict[str, Any],
        skills_by_category: dict[str, list[str]],
        readability: dict[str, Any]
    ) -> dict[str, Any]:
        """Compute full ATS compliance score and diagnostics."""

        scores: dict[str, float] = {}
        strengths: list[str] = []
        weaknesses: list[str] = []
        recommendations: list[str] = []

        # 1. Contact Info (Max 10)
        c_score = 0.0
        if contact_info.get("email"):
            c_score += 4.0
        else:
            weaknesses.append("Missing or unparseable email address.")
            recommendations.append("Add a standard professional email address in the header.")

        if contact_info.get("phone"):
            c_score += 3.0
        else:
            weaknesses.append("Missing phone number.")
            recommendations.append("Include your contact phone number with country/area code.")

        if contact_info.get("linkedin") or contact_info.get("github"):
            c_score += 3.0
            strengths.append("Professional profile links (LinkedIn/GitHub) present.")
        else:
            weaknesses.append("Missing professional portfolio or LinkedIn URL.")
            recommendations.append("Add your LinkedIn and GitHub/portfolio profiles.")
        scores["Contact Info"] = round(c_score, 1)

        # 2. Section Structure (Max 15)
        s_score = 0.0
        essential_sections = ["experience", "education", "skills"]
        bonus_sections = ["summary", "projects", "certifications"]

        for sec in essential_sections:
            if sections.get(sec) and len(sections[sec].split()) > 5:
                s_score += 3.5
            else:
                weaknesses.append(f"Missing or unclear '{sec.title()}' section header.")
                recommendations.append(f"Add a distinct, clear heading for '{sec.title()}'.")

        for sec in bonus_sections:
            if sections.get(sec) and len(sections[sec].split()) > 5:
                s_score += 1.5

        s_score = min(15.0, s_score)
        if s_score >= 12.0:
            strengths.append("Well-structured document with clear standard ATS section headings.")
        scores["Section Structure"] = round(s_score, 1)

        # 3. Action Verbs (Max 15)
        verb_ratio = bullet_analysis.get("action_verb_ratio", 0.0)
        if verb_ratio >= 75.0:
            v_score = 15.0
            strengths.append(f"Strong active voice: {verb_ratio}% of bullet points begin with power action verbs.")
        elif verb_ratio >= 50.0:
            v_score = 11.0
            strengths.append(f"Good active voice usage ({verb_ratio}%).")
        elif verb_ratio >= 25.0:
            v_score = 7.0
            weaknesses.append(f"Only {verb_ratio}% of bullet points use strong action verbs.")
            recommendations.append("Start bullet points with high-impact verbs (e.g., 'Architected', 'Spearheaded', 'Optimized').")
        else:
            v_score = 3.0
            weaknesses.append("Rarely uses power action verbs at the beginning of experience bullets.")
            recommendations.append("Rewrite job descriptions using the Action-Verb + Task + Result framework.")
        scores["Action Verbs"] = round(v_score, 1)

        # 4. Quantifiable Metrics (Max 20)
        metric_ratio = bullet_analysis.get("quantifiable_ratio", 0.0)
        if metric_ratio >= 50.0:
            m_score = 20.0
            strengths.append(f"Excellent business impact: {metric_ratio}% of bullet points contain quantifiable numbers/metrics.")
        elif metric_ratio >= 30.0:
            m_score = 15.0
            strengths.append(f"Solid quantifiable evidence present ({metric_ratio}%).")
        elif metric_ratio >= 15.0:
            m_score = 10.0
            weaknesses.append(f"Only {metric_ratio}% of bullets have numbers or quantifiable metrics.")
            recommendations.append("Add measurable achievements (e.g. '% latency reduction', '$ revenue saved', 'user growth').")
        else:
            m_score = 4.0
            weaknesses.append("Lacks quantifiable impact and numerical results.")
            recommendations.append("Incorporate specific metrics (e.g. 'Reduced CI/CD build times by 45%', 'Scaled service to 10k QPS').")
        scores["Quantifiable Metrics"] = round(m_score, 1)

        # 5. Technical Depth & Breadth (Max 20)
        total_skills = sum(len(skills) for skills in skills_by_category.values())
        categories_count = len(skills_by_category)

        if total_skills >= 18 and categories_count >= 4:
            t_score = 20.0
            strengths.append(f"Broad technical repertoire: {total_skills} verified technical skills across {categories_count} domains.")
        elif total_skills >= 12 and categories_count >= 3:
            t_score = 16.0
            strengths.append(f"Healthy skill diversity ({total_skills} technical skills detected).")
        elif total_skills >= 6:
            t_score = 11.0
            weaknesses.append(f"Moderate skill breadth ({total_skills} skills found).")
            recommendations.append("Include specific tools, libraries, and cloud platforms explicitly in a dedicated Skills section.")
        else:
            t_score = 5.0
            weaknesses.append("Low detected technical skill density.")
            recommendations.append("Add an explicit Skills matrix divided into Languages, Frameworks, Cloud, and Databases.")
        scores["Technical Depth"] = round(t_score, 1)

        # 6. Readability & Formatting (Max 10)
        ease = readability.get("flesch_reading_ease", 50.0)
        words_per_sent = readability.get("words_per_sentence", 15.0)

        r_score = 0.0
        if 40.0 <= ease <= 75.0:
            r_score += 5.0
            strengths.append(f"Optimal professional readability score ({ease}/100).")
        elif ease < 40.0:
            r_score += 2.5
            weaknesses.append("Text is overly complex or dense with lengthy run-on sentences.")
            recommendations.append("Shorten complex sentences to improve readability for recruiters.")
        else:
            r_score += 4.0

        if 10.0 <= words_per_sent <= 22.0:
            r_score += 5.0
        else:
            r_score += 2.5
            recommendations.append("Aim for 12-20 words per bullet point for punchy, impactful reading.")
        scores["Readability & Layout"] = round(r_score, 1)

        # 7. Length and Density (Max 10)
        total_words = readability.get("total_words", 0)
        if 350 <= total_words <= 900:
            l_score = 10.0
            strengths.append(f"Ideal document length ({total_words} words).")
        elif 250 <= total_words < 350:
            l_score = 7.0
            weaknesses.append("Resume is somewhat brief; consider detailing project impacts.")
        elif 900 < total_words <= 1400:
            l_score = 7.5
            weaknesses.append("Resume is getting lengthy (may exceed 2 pages). Consider condensing older roles.")
        elif total_words > 1400:
            l_score = 4.0
            weaknesses.append(f"Resume is excessively long ({total_words} words). Risk of recruiter fatigue.")
            recommendations.append("Condense content to 1-2 pages maximum by focusing only on high-impact accomplishments.")
        else:
            l_score = 3.0
            weaknesses.append("Resume has very little content; expand on technical experiences.")
        scores["Length & Density"] = round(l_score, 1)

        # Total Score
        total_score = round(sum(scores.values()), 1)
        total_score = max(0.0, min(100.0, total_score))

        # Assign Grade
        if total_score >= 90:
            grade = "A+ (Elite ATS Pass)"
        elif total_score >= 80:
            grade = "A (High Interview Rate)"
        elif total_score >= 70:
            grade = "B (Competitive, Needs Polish)"
        elif total_score >= 55:
            grade = "C (Average, Risks ATS Rejection)"
        else:
            grade = "D (Urgent Optimization Needed)"

        return {
            "total_score": total_score,
            "grade": grade,
            "category_scores": scores,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendations": recommendations,
        }
