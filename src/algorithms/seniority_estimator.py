"""
Seniority Estimator and Career Progression Roadmap Engine
Predicts professional engineering tier, market compensation bands, and growth milestones.
"""

import re
from typing import Any, ClassVar

from src.config import POWER_ACTION_VERBS


class SeniorityEstimator:
    """Estimates career seniority level, compensation benchmark, and career advancement roadmap."""

    SALARY_BANDS: ClassVar[dict[str, dict[str, str]]] = {
        "Machine Learning & AI": {
            "Junior": "$115,000 - $145,000",
            "Mid-Level": "$145,000 - $185,000",
            "Senior": "$185,000 - $240,000",
            "Staff / Lead": "$240,000 - $320,000",
            "Principal": "$320,000 - $450,000+"
        },
        "Full-Stack & Backend": {
            "Junior": "$95,000 - $125,000",
            "Mid-Level": "$125,000 - $165,000",
            "Senior": "$165,000 - $215,000",
            "Staff / Lead": "$215,000 - $280,000",
            "Principal": "$280,000 - $380,000+"
        },
        "DevOps & Cloud Infrastructure": {
            "Junior": "$100,000 - $130,000",
            "Mid-Level": "$130,000 - $170,000",
            "Senior": "$170,000 - $225,000",
            "Staff / Lead": "$225,000 - $295,000",
            "Principal": "$295,000 - $400,000+"
        },
        "Data Engineering & Analytics": {
            "Junior": "$100,000 - $130,000",
            "Mid-Level": "$130,000 - $165,000",
            "Senior": "$165,000 - $215,000",
            "Staff / Lead": "$215,000 - $285,000",
            "Principal": "$285,000 - $390,000+"
        }
    }

    CAREER_ROADMAPS: ClassVar[dict[str, dict[str, Any]]] = {
        "Junior": {
            "next_level": "Mid-Level Engineer",
            "technical_milestones": [
                "Master system design fundamentals and asynchronous API patterns.",
                "Own end-to-end features from requirement gathering to production deployment.",
                "Deepen knowledge of database indexing, query optimization, and caching."
            ],
            "leadership_milestones": [
                "Participate actively in code reviews and write thorough unit/integration tests.",
                "Improve technical documentation and contribute to team runbooks."
            ]
        },
        "Mid-Level": {
            "next_level": "Senior Engineer",
            "technical_milestones": [
                "Architect scalable microservices and design distributed event-driven systems.",
                "Drive performance benchmarking, latency profiling, and cost optimization.",
                "Implement robust observability (metrics, logs, distributed tracing)."
            ],
            "leadership_milestones": [
                "Mentor junior engineers and lead technical sprint design reviews.",
                "Align cross-functional project deliverables with product stakeholders."
            ]
        },
        "Senior": {
            "next_level": "Staff / Principal Architect",
            "technical_milestones": [
                "Design enterprise-scale multi-region architectures with 99.99%+ availability.",
                "Standardize architectural blueprints and engineering patterns across multiple teams.",
                "Drive technical debt reduction and long-term technology radar evaluation."
            ],
            "leadership_milestones": [
                "Champion engineering culture, hire senior talent, and influence technical vision.",
                "Advise executive leadership on infrastructure investments and ROI."
            ]
        },
        "Staff / Lead": {
            "next_level": "Principal Engineer / Fellow",
            "technical_milestones": [
                "Pioneer proprietary internal platforms and patented core algorithmic frameworks.",
                "Drive company-wide reliability, security posture, and multi-cloud strategies."
            ],
            "leadership_milestones": [
                "Set 3-5 year technical strategy and represent the company in industry consortia.",
                "Guide director-level engineering roadmaps and large-scale organizational transformations."
            ]
        }
    }

    @classmethod
    def estimate_seniority(
        cls,
        resume_text: str,
        predicted_domain: str,
        bullet_analysis: dict[str, any]
    ) -> dict[str, any]:
        """
        Estimate seniority level, compensation benchmark, and career advancement roadmap.
        """
        # 1. Year Span Analysis (detect years like 2017 - Present, 2019 - 2022)
        year_matches = re.findall(r"\b(20[0-2][0-9]|199[0-9])\b", resume_text)
        years = [int(y) for y in year_matches]
        
        estimated_years_exp = 3.0  # default baseline
        if len(years) >= 2:
            min_year = min(years)
            max_year = 2026
            span = max_year - min_year
            estimated_years_exp = max(1.0, min(25.0, span))

        # Check explicit "X+ years of experience" statements
        explicit_yoe = re.search(r"(\d+)\+?\s*(?:years|yrs)(?:\s*of)?\s*experience", resume_text, re.IGNORECASE)
        if explicit_yoe:
            estimated_years_exp = max(estimated_years_exp, float(explicit_yoe.group(1)))

        # 2. Leadership & System Design Indicators
        leadership_verbs = POWER_ACTION_VERBS["leadership"]
        leadership_count = sum(
            1 for b in bullet_analysis.get("bullet_details", [])
            if b.get("detected_verb") in leadership_verbs
        )

        has_staff_signals = bool(re.search(r"\b(staff|principal|lead|director|architect|head of|spearheaded|governed)\b", resume_text, re.IGNORECASE))
        has_senior_signals = bool(re.search(r"\b(senior|sr\.|mentored|architected|scaled|optimized)\b", resume_text, re.IGNORECASE))

        # Determine Tier
        if estimated_years_exp >= 10 and leadership_count >= 3 and has_staff_signals:
            level = "Principal"
        elif estimated_years_exp >= 7 and (leadership_count >= 2 or has_staff_signals):
            level = "Staff / Lead"
        elif estimated_years_exp >= 4 or has_senior_signals:
            level = "Senior"
        elif estimated_years_exp >= 2:
            level = "Mid-Level"
        else:
            level = "Junior"

        # Lookup salary band
        domain_salaries = cls.SALARY_BANDS.get(predicted_domain, cls.SALARY_BANDS["Full-Stack & Backend"])
        salary_range = domain_salaries.get(level, "$140,000 - $190,000")

        # Lookup roadmap
        roadmap_key = level if level in cls.CAREER_ROADMAPS else "Senior"
        roadmap = cls.CAREER_ROADMAPS.get(roadmap_key, cls.CAREER_ROADMAPS["Senior"])

        return {
            "seniority_level": level,
            "estimated_years_experience": round(estimated_years_exp, 1),
            "leadership_bullet_count": leadership_count,
            "salary_benchmark": salary_range,
            "next_level": roadmap["next_level"],
            "technical_milestones": roadmap["technical_milestones"],
            "leadership_milestones": roadmap["leadership_milestones"]
        }
