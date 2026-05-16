"""Custom CSS styles for the FaceVision AI Streamlit app."""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* ── Global ──────────────────────────────────────────────────────────────── */
* { font-family: 'Outfit', sans-serif; }
.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #0d1127 40%, #0a0a1a 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1127 0%, #161638 100%);
    border-right: 1px solid rgba(0,212,255,0.15);
}
[data-testid="stSidebar"] .stMarkdown h1 {
    background: linear-gradient(135deg, #00d4ff, #7b2fff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-weight: 800; font-size: 1.6rem; text-align: center;
}
[data-testid="stSidebar"] .stRadio > label {
    color: #8892b0 !important; font-weight: 500;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px; padding: 10px 16px; margin-bottom: 6px;
    transition: all 0.3s ease;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: rgba(0,212,255,0.08); border-color: rgba(0,212,255,0.3);
    transform: translateX(4px);
}

/* ── Cards ───────────────────────────────────────────────────────────────── */
.glass-card {
    background: rgba(255,255,255,0.03); backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.08); border-radius: 16px;
    padding: 24px; margin-bottom: 20px;
    transition: all 0.3s ease;
}
.glass-card:hover {
    border-color: rgba(0,212,255,0.25);
    box-shadow: 0 8px 32px rgba(0,212,255,0.08);
}

/* ── Hero Section ────────────────────────────────────────────────────────── */
.hero-section {
    text-align: center; padding: 60px 20px 40px;
}
.hero-title {
    font-size: 3.2rem; font-weight: 800; line-height: 1.1; margin-bottom: 16px;
    background: linear-gradient(135deg, #00d4ff 0%, #7b2fff 50%, #ff6b9d 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: shimmer 3s ease-in-out infinite alternate;
}
@keyframes shimmer {
    0% { filter: hue-rotate(0deg); } 100% { filter: hue-rotate(20deg); }
}
.hero-subtitle {
    font-size: 1.15rem; color: #8892b0; max-width: 700px;
    margin: 0 auto; line-height: 1.7;
}

/* ── Metric Cards ────────────────────────────────────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(123,47,255,0.08));
    border: 1px solid rgba(0,212,255,0.15); border-radius: 14px;
    padding: 20px; text-align: center;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0,212,255,0.12);
}
.metric-value {
    font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(135deg, #00d4ff, #7b2fff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.metric-label { font-size: 0.85rem; color: #8892b0; margin-top: 4px; font-weight: 500; }

/* ── Feature Cards ───────────────────────────────────────────────────────── */
.feature-card {
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px; padding: 28px; text-align: center;
    transition: all 0.3s ease; min-height: 180px;
}
.feature-card:hover {
    background: rgba(0,212,255,0.05); border-color: rgba(0,212,255,0.2);
    transform: translateY(-6px); box-shadow: 0 16px 48px rgba(0,212,255,0.1);
}
.feature-icon { font-size: 2.5rem; margin-bottom: 12px; }
.feature-title {
    font-size: 1.1rem; font-weight: 700; color: #e0e0e0; margin-bottom: 8px;
}
.feature-desc { font-size: 0.85rem; color: #8892b0; line-height: 1.5; }

/* ── Section Headers ─────────────────────────────────────────────────────── */
.section-header {
    font-size: 1.8rem; font-weight: 700; margin-bottom: 8px;
    background: linear-gradient(135deg, #00d4ff, #7b2fff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.section-sub { color: #8892b0; font-size: 0.95rem; margin-bottom: 24px; }

/* ── Results / Status ────────────────────────────────────────────────────── */
.result-badge {
    display: inline-block; padding: 6px 16px; border-radius: 20px;
    font-weight: 600; font-size: 0.85rem;
}
.badge-success { background: rgba(0,255,136,0.12); color: #00ff88; border: 1px solid rgba(0,255,136,0.3); }
.badge-danger  { background: rgba(255,69,58,0.12);  color: #ff453a; border: 1px solid rgba(255,69,58,0.3); }
.badge-info    { background: rgba(0,212,255,0.12);  color: #00d4ff; border: 1px solid rgba(0,212,255,0.3); }

/* ── Comparison Bar ──────────────────────────────────────────────────────── */
.similarity-bar-bg {
    background: rgba(255,255,255,0.05); border-radius: 12px;
    height: 28px; width: 100%; overflow: hidden; margin: 12px 0;
}
.similarity-bar-fill {
    height: 100%; border-radius: 12px;
    background: linear-gradient(90deg, #ff453a, #ff9500, #00ff88);
    transition: width 1s ease;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.8rem; color: #0a0a1a;
}

/* ── Tabs ─────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px; background: transparent;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px; padding: 8px 20px; color: #8892b0;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,212,255,0.1) !important;
    border-color: rgba(0,212,255,0.3) !important; color: #00d4ff !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #7b2fff) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    padding: 10px 28px !important; transition: all 0.3s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(0,212,255,0.3) !important;
}

/* ── File Uploader ───────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(0,212,255,0.2) !important; border-radius: 14px;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,212,255,0.4) !important;
}

/* ── Divider ─────────────────────────────────────────────────────────────── */
.styled-divider {
    height: 1px; margin: 32px 0;
    background: linear-gradient(90deg, transparent, rgba(0,212,255,0.3), transparent);
}

/* ── Person Card ─────────────────────────────────────────────────────────── */
.person-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 16px; text-align: center;
    transition: all 0.3s ease;
}
.person-card:hover {
    border-color: rgba(0,212,255,0.25); transform: translateY(-2px);
}
.person-name {
    font-weight: 700; font-size: 1rem; color: #00d4ff; margin-top: 8px;
}
.person-count { font-size: 0.8rem; color: #8892b0; }

/* ── Emotion Chart Area ──────────────────────────────────────────────────── */
.analysis-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px; margin-top: 16px;
}
.attr-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 16px; text-align: center;
}
.attr-value { font-size: 1.4rem; font-weight: 700; color: #00d4ff; }
.attr-label { font-size: 0.78rem; color: #8892b0; margin-top: 4px; }
</style>
"""
