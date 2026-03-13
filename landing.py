"""
SafeRoute AI — Landing Page

A clean, modern landing page that introduces the project and links
to the main Streamlit dashboard.

Run with: streamlit run landing.py
"""

import streamlit as st

st.set_page_config(
    page_title="SafeRoute AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Hide default Streamlit chrome for a cleaner landing feel
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ---------- reset ---------- */
    [data-testid="stSidebar"] { display: none; }
    header[data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 0 !important; max-width: 1200px; }

    /* ---------- hero ---------- */
    .hero {
        text-align: center;
        padding: 80px 20px 40px;
        background: linear-gradient(160deg, #0a0a1a 0%, #101030 50%, #0d1f3c 100%);
        border-radius: 0 0 32px 32px;
        margin: -1rem -1rem 0;
    }
    .hero h1 {
        font-size: 56px; font-weight: 800; margin: 0;
        background: linear-gradient(135deg, #64b5f6, #00e5ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero .tagline {
        font-size: 20px; color: #b0bec5; margin: 16px auto 32px;
        max-width: 640px; line-height: 1.6;
    }
    .hero .cta-btn {
        display: inline-block;
        padding: 16px 48px;
        font-size: 18px; font-weight: 700;
        color: #fff;
        background: linear-gradient(135deg, #1e88e5, #00b0ff);
        border-radius: 50px;
        text-decoration: none;
        box-shadow: 0 6px 24px rgba(0,176,255,0.35);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .hero .cta-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 32px rgba(0,176,255,0.5);
    }

    /* ---------- features ---------- */
    .features {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 24px;
        padding: 60px 20px 40px;
    }
    .feature-card {
        background: #12122a;
        border: 1px solid #1e1e40;
        border-radius: 16px;
        padding: 32px 24px;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 28px rgba(0,0,0,0.4);
    }
    .feature-icon { font-size: 42px; margin-bottom: 12px; }
    .feature-card h3 {
        font-size: 18px; color: #e0e0e0; margin: 0 0 8px;
    }
    .feature-card p {
        font-size: 14px; color: #888; margin: 0; line-height: 1.5;
    }

    /* ---------- how-it-works ---------- */
    .pipeline-section {
        text-align: center;
        padding: 40px 20px 60px;
    }
    .pipeline-section h2 {
        font-size: 28px; color: #cfd8dc; margin-bottom: 24px;
    }
    .pipeline-flow {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        font-size: 16px;
        color: #90a4ae;
    }
    .pipeline-flow .step {
        background: #1a1a35;
        padding: 10px 20px;
        border-radius: 8px;
        color: #64b5f6;
        font-weight: 600;
    }
    .pipeline-flow .arrow { color: #455a64; font-size: 20px; }

    /* ---------- footer ---------- */
    .footer {
        text-align: center;
        padding: 32px 20px;
        color: #555;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Hero section
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <p style="font-size:64px;margin:0 0 8px;">🛡️</p>
    <h1>SafeRoute AI</h1>
    <p class="tagline">
        AI-powered safe navigation system that predicts urban risk
        and suggests safer routes.
    </p>
</div>
""", unsafe_allow_html=True)

# CTA button — uses Streamlit's native page switching
if st.button("Open Safety Map", type="primary", use_container_width=False):
    st.switch_page("pages/dashboard.py")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Features section
# ---------------------------------------------------------------------------
st.markdown("""
<div class="features">
    <div class="feature-card">
        <div class="feature-icon">🤖</div>
        <h3>AI Risk Prediction</h3>
        <p>Machine-learning model trained on crowd, lighting, time, and crime data to score road-segment safety from 0 to 10.</p>
    </div>
    <div class="feature-card">
        <div class="feature-icon">👥</div>
        <h3>Crowd Detection</h3>
        <p>YOLOv8 computer vision detects people in real-time video and classifies crowd density as low, medium, or high.</p>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🗺️</div>
        <h3>Safety Heatmap</h3>
        <p>Interactive map of Delhi showing color-coded safety markers — green for safe, yellow for moderate, red for high risk.</p>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🧭</div>
        <h3>Safe Route Navigation</h3>
        <p>Dijkstra's algorithm finds the safest path between locations using cost&nbsp;=&nbsp;distance&nbsp;+&nbsp;risk.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# How it works
# ---------------------------------------------------------------------------
st.markdown("""
<div class="pipeline-section">
    <h2>How It Works</h2>
    <div class="pipeline-flow">
        <span class="step">📹 Video</span>
        <span class="arrow">→</span>
        <span class="step">👥 Crowd Detection</span>
        <span class="arrow">→</span>
        <span class="step">🔬 Feature Extraction</span>
        <span class="arrow">→</span>
        <span class="step">🤖 Risk Model</span>
        <span class="arrow">→</span>
        <span class="step">📊 Safety Score</span>
        <span class="arrow">→</span>
        <span class="step">🧭 Routing</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
<div class="footer">
    SafeRoute AI — Prototype &nbsp;|&nbsp; Built with YOLOv8 · scikit-learn · Streamlit
</div>
""", unsafe_allow_html=True)
