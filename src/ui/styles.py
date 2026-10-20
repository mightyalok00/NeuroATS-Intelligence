"""
Enterprise Design System and Custom CSS Tokens for AdaptiveML
"""

ENTERPRISE_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --font-main: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-mono: 'JetBrains Mono', monospace;
        --bg-slate-900: #0F172A;
        --bg-slate-800: #1E293B;
        --bg-slate-700: #334155;
        --accent-indigo: #6366F1;
        --accent-purple: #8B5CF6;
        --accent-pink: #EC4899;
        --accent-emerald: #10B981;
        --accent-amber: #F59E0B;
        --accent-rose: #F43F5E;
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --border-glass: rgba(255, 255, 255, 0.08);
    }

    html, body, [class*="css"] {
        font-family: var(--font-main);
    }

    /* Modern Application Header */
    .app-header-container {
        padding: 1.2rem 0 1.8rem 0;
        border-bottom: 1px solid var(--border-glass);
        margin-bottom: 1.8rem;
    }
    .brand-title {
        background: linear-gradient(135deg, #818CF8 0%, #C084FC 50%, #F472B6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.2rem;
        letter-spacing: -0.02em;
        margin: 0;
        line-height: 1.2;
    }
    .brand-subtitle {
        color: var(--text-secondary);
        font-size: 0.95rem;
        font-weight: 400;
        margin-top: 0.35rem;
    }
    .version-tag {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #A5B4FC;
        font-size: 0.75rem;
        font-family: var(--font-mono);
        padding: 2px 8px;
        border-radius: 9999px;
        font-weight: 600;
        margin-left: 0.5rem;
        vertical-align: middle;
    }

    /* KPI Glassmorphism Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid var(--border-glass);
        border-radius: 14px;
        padding: 18px 22px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
    }
    .kpi-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-secondary);
        font-weight: 600;
        margin-bottom: 6px;
    }
    .kpi-metric {
        font-size: 2.1rem;
        font-weight: 800;
        color: var(--text-primary);
        line-height: 1.15;
        letter-spacing: -0.02em;
    }
    .kpi-caption {
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 6px;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .text-emerald { color: #34D399; }
    .text-indigo  { color: #818CF8; }
    .text-amber   { color: #FBBF24; }
    .text-purple  { color: #C084FC; }

    /* Custom Skill Badges */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        padding: 3px 11px;
        margin: 3px 5px 3px 0;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 500;
        letter-spacing: 0.01em;
    }
    .badge-matched {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #6EE7B7;
    }
    .badge-missing {
        background: rgba(244, 63, 94, 0.12);
        border: 1px solid rgba(244, 63, 94, 0.35);
        color: #FDA4AF;
    }
    .badge-keyword {
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.35);
        color: #C7D2FE;
    }

    /* Section Subheadings */
    .view-heading {
        font-size: 1.35rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.25rem;
        letter-spacing: -0.01em;
    }
    .view-sub {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin-bottom: 1.25rem;
    }

    /* Chat Badge */
    .intent-indicator {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 102, 241, 0.18);
        border: 1px solid rgba(99, 102, 241, 0.4);
        color: #E0E7FF;
        padding: 5px 14px;
        border-radius: 8px;
        font-size: 0.84rem;
        font-weight: 600;
        font-family: var(--font-mono);
        margin-bottom: 14px;
    }

    /* Interactive Filter Bar & Status Badges */
    .filter-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 10px;
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 10px 18px;
        margin-bottom: 1.4rem;
    }
    .filter-group {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }
    .filter-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        font-family: var(--font-mono);
    }
    .filter-pill-green {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #6EE7B7;
    }
    .filter-pill-amber {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.4);
        color: #FCD34D;
    }
    .filter-pill-indigo {
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.4);
        color: #C7D2FE;
    }
    .filter-pill-purple {
        background: rgba(168, 85, 247, 0.15);
        border: 1px solid rgba(168, 85, 247, 0.4);
        color: #D8B4FE;
    }

    /* Active Resume Profile Card & Upload Component */
    .active-profile-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 14px;
        padding: 12px 20px;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    .active-profile-meta {
        display: flex;
        flex-direction: column;
        gap: 2px;
    }
    .active-profile-label {
        font-size: 0.75rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
        font-family: var(--font-mono);
    }
    .active-profile-name {
        font-size: 1.15rem;
        font-weight: 700;
        color: #F8FAFC;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .active-profile-badge {
        font-size: 0.72rem;
        padding: 3px 10px;
        border-radius: 9999px;
        font-family: var(--font-mono);
        font-weight: 600;
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid rgba(99, 102, 241, 0.4);
        color: #C7D2FE;
    }
    .active-profile-badge.uploaded {
        background: rgba(16, 185, 129, 0.2);
        border-color: rgba(16, 185, 129, 0.5);
        color: #6EE7B7;
    }
</style>
"""
