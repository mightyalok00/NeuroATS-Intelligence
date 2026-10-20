"""
System Configurations, Rubrics, and Taxonomy Dictionaries
"""

APP_TITLE = "NeuroATS Intelligence Platform"
APP_SUBTITLE = "Local-First AI & Machine Learning Engine for ATS Analytics, Candidate Ranking & Adaptive Q&A"
VERSION = "2.1.0"
APP_URL = "https://neuroats-intelligence.streamlit.app/"
REPO_URL = "https://github.com/mightyalok00/NeuroATS-Intelligence"

# Domains recognized by the Supervised Domain Classifier
DOMAINS = [
    "Machine Learning & AI",
    "Full-Stack & Backend",
    "DevOps & Cloud Infrastructure",
    "Data Engineering & Analytics",
    "Frontend & Mobile",
    "Cybersecurity & Systems",
    "Product & Technical Management",
]

# Query Intents for the Adaptive ML Router
INTENT_CLASSES = [
    "SPECIFIC_DETAIL",      # Fact retrieval from resume (skills, companies, education)
    "ATS_DIAGNOSTICS",      # ATS score, resume critique, formatting feedback
    "JOB_FIT_MATCH",        # Comparison against a job description or target role
    "INTERVIEW_PREP",       # Generation of interview questions based on experience
    "EXECUTIVE_SUMMARY",    # High-level overview, career trajectory, impact
]

# High-impact Action Verbs across resume engineering
POWER_ACTION_VERBS = {
    "leadership": [
        "orchestrated", "spearheaded", "championed", "mentored", "directed",
        "founded", "governed", "steered", "empowered", "cultivated", "mobilized"
    ],
    "engineering": [
        "architected", "engineered", "implemented", "developed", "deployed",
        "refactored", "optimized", "benchmarked", "automated", "containerized",
        "migrated", "scaled", "debugged", "designed", "integrated", "constructed"
    ],
    "impact_and_results": [
        "accelerated", "amplified", "boosted", "maximized", "curtailed",
        "reduced", "eliminated", "generated", "yielded", "elevated", "enhanced",
        "surpassed", "expanded", "outperformed", "transformed"
    ],
    "analytics_and_research": [
        "synthesized", "evaluated", "modeled", "forecasted", "diagnosed",
        "investigated", "analyzed", "validated", "quantified", "discovered"
    ]
}

# Flat set of all power verbs
ALL_ACTION_VERBS = {
    verb for cat_verbs in POWER_ACTION_VERBS.values() for verb in cat_verbs
}

# ATS Scoring Rubric Weighting
ATS_WEIGHTS = {
    "contact_info": 10,        # Email, phone, linkedin/github links
    "section_structure": 15,   # Essential sections detected
    "action_verbs": 15,        # Ratio of bullets beginning with power verbs
    "quantifiable_metrics": 20,# Bullets containing numbers/percentages/dollars
    "technical_depth": 20,     # Depth and breadth of detected technical skills
    "readability_format": 10,  # Sentence length, word count, flesch readability
    "length_and_density": 10,  # Appropriate length (not too short or bloated)
}
