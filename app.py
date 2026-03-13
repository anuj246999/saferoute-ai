"""
SafeRoute AI — Streamlit Visual Dashboard

A visual interface for the SafeRoute AI pipeline that displays:
- Crowd density detection with bounding boxes
- Predicted risk score with color-coded safety status
- Map-like visualization of location safety
- Step-by-step pipeline progress

Run with: streamlit run app.py
"""

import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from vision.crowd_detector import CrowdDetector, classify_density, density_to_numeric
from models.feature_extractor import extract_features, get_historical_crime_score
from models.risk_model import generate_sample_dataset, train_model, load_model, predict_risk
from routing.dijkstra import create_sample_graph

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SafeRoute AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for visual indicators
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .safe-box {
        background: linear-gradient(135deg, #00c853, #69f0ae);
        color: white; padding: 20px; border-radius: 12px;
        text-align: center; font-size: 24px; font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,200,83,0.3);
    }
    .moderate-box {
        background: linear-gradient(135deg, #ff9800, #ffcc02);
        color: white; padding: 20px; border-radius: 12px;
        text-align: center; font-size: 24px; font-weight: bold;
        box-shadow: 0 4px 15px rgba(255,152,0,0.3);
    }
    .danger-box {
        background: linear-gradient(135deg, #f44336, #ff5252);
        color: white; padding: 20px; border-radius: 12px;
        text-align: center; font-size: 24px; font-weight: bold;
        box-shadow: 0 4px 15px rgba(244,67,54,0.3);
    }
    .metric-card {
        background: #1e1e2f; color: white; padding: 18px;
        border-radius: 10px; text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    .metric-card h3 { margin: 0; font-size: 14px; opacity: 0.7; }
    .metric-card h1 { margin: 5px 0 0 0; font-size: 32px; }
    .pipeline-step {
        background: #262640; color: #ccc; padding: 12px 18px;
        border-radius: 8px; margin-bottom: 6px; font-size: 15px;
        border-left: 4px solid #555;
    }
    .pipeline-step.active {
        background: #1a3a5c; color: #64b5f6;
        border-left: 4px solid #42a5f5;
    }
    .pipeline-step.done {
        background: #1a3c1a; color: #81c784;
        border-left: 4px solid #66bb6a;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_safety_status(risk_score: float) -> tuple[str, str, str]:
    """Return (label, css_class, color) based on risk score."""
    if risk_score <= 3.5:
        return "SAFE", "safe-box", "#00c853"
    elif risk_score <= 6.5:
        return "MODERATE RISK", "moderate-box", "#ff9800"
    else:
        return "HIGH RISK", "danger-box", "#f44336"


def draw_bounding_boxes(frame: np.ndarray, boxes: list[dict], density: str) -> np.ndarray:
    """Draw bounding boxes on frame with color based on density."""
    color_map = {"low": (0, 200, 0), "medium": (0, 200, 255), "high": (0, 0, 255)}
    color = color_map.get(density, (0, 200, 0))
    annotated = frame.copy()
    for box in boxes:
        cv2.rectangle(annotated, (box["x1"], box["y1"]), (box["x2"], box["y2"]), color, 2)
        label = f"person {box['confidence']:.0%}"
        cv2.putText(annotated, label, (box["x1"], box["y1"] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return annotated


def render_metric(label: str, value: str) -> str:
    """Return HTML for a dark metric card."""
    return f'<div class="metric-card"><h3>{label}</h3><h1>{value}</h1></div>'


def pipeline_step_html(text: str, state: str = "pending") -> str:
    """Return HTML for a pipeline progress step."""
    icons = {"pending": "⏳", "active": "🔄", "done": "✅"}
    icon = icons.get(state, "⏳")
    return f'<div class="pipeline-step {state}">{icon}  {text}</div>'


LOCATIONS = {
    "road_segment_1": {"name": "Main Street", "lat": 28.6139, "lon": 77.2090},
    "road_segment_3": {"name": "Park Avenue", "lat": 28.6180, "lon": 77.2150},
    "road_segment_5": {"name": "Station Road", "lat": 28.6100, "lon": 77.2050},
    "road_segment_7": {"name": "Dark Alley", "lat": 28.6060, "lon": 77.2110},
    "road_segment_12": {"name": "Highway 12", "lat": 28.6200, "lon": 77.2000},
}


# ---------------------------------------------------------------------------
# Sidebar configuration
# ---------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/route.png", width=64)
st.sidebar.title("SafeRoute AI")
st.sidebar.markdown("---")

location_key = st.sidebar.selectbox(
    "Select Road Segment",
    list(LOCATIONS.keys()),
    format_func=lambda k: f"{LOCATIONS[k]['name']} ({k})",
)
location_info = LOCATIONS[location_key]

sim_hour = st.sidebar.slider("Simulate Hour of Day", 0, 23, 23)
sim_crowd = st.sidebar.selectbox("Simulate Crowd Density", ["low", "medium", "high"])

use_video = st.sidebar.checkbox("Run Crowd Detection from Video", value=False)

st.sidebar.markdown("---")
st.sidebar.caption("SafeRoute AI v1.0 — Prototype")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title("🛡️ SafeRoute AI — Urban Safety Dashboard")
st.caption("Real-time AI pipeline: video → crowd detection → risk prediction → safe routing")

# Pipeline progress placeholder
pipeline_placeholder = st.empty()

def update_pipeline(steps: dict[str, str]) -> None:
    """Render the pipeline progress bar."""
    html = '<div style="margin-bottom:18px;">'
    for text, state in steps.items():
        html += pipeline_step_html(text, state)
    html += "</div>"
    pipeline_placeholder.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Run pipeline button
# ---------------------------------------------------------------------------
if st.button("▶  Run SafeRoute AI Pipeline", type="primary", use_container_width=True):

    # -- Ensure model is trained --------------------------------------------------
    steps = {
        "Initializing model...": "active",
        "Crowd Detection (YOLOv8)": "pending",
        "Feature Extraction": "pending",
        "Risk Prediction (ML)": "pending",
        "Route Optimization (Dijkstra)": "pending",
    }
    update_pipeline(steps)
    time.sleep(0.4)

    model = load_model()
    steps["Initializing model..."] = "done"

    # -- Crowd detection ----------------------------------------------------------
    steps["Crowd Detection (YOLOv8)"] = "active"
    update_pipeline(steps)

    detected_frame = None
    boxes = []

    if use_video:
        video_path = os.path.join(os.path.dirname(__file__), "data", "sample_video.avi")
        if not os.path.exists(video_path):
            from data.generate_sample_video import generate_sample_video
            generate_sample_video(output_path=video_path)

        detector = CrowdDetector(model_path="yolov8n.pt", confidence=0.4)
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if ret:
            crowd_info = detector.get_crowd_density(frame)
            crowd_density_label = crowd_info["density"]
            crowd_density_num = crowd_info["density_numeric"]
            person_count = crowd_info["person_count"]
            boxes = crowd_info["boxes"]
            detected_frame = draw_bounding_boxes(frame, boxes, crowd_density_label)
        else:
            crowd_density_label = sim_crowd
            crowd_density_num = density_to_numeric(sim_crowd)
            person_count = 0
    else:
        crowd_density_label = sim_crowd
        crowd_density_num = density_to_numeric(sim_crowd)
        person_count = {"low": 2, "medium": 10, "high": 22}[sim_crowd]

    time.sleep(0.3)
    steps["Crowd Detection (YOLOv8)"] = "done"

    # -- Feature extraction -------------------------------------------------------
    steps["Feature Extraction"] = "active"
    update_pipeline(steps)

    timestamp = datetime(2026, 3, 13, sim_hour, 30)
    features = extract_features(
        crowd_density=crowd_density_num,
        location=location_key,
        timestamp=timestamp,
    )
    time.sleep(0.3)
    steps["Feature Extraction"] = "done"

    # -- Risk prediction ----------------------------------------------------------
    steps["Risk Prediction (ML)"] = "active"
    update_pipeline(steps)

    risk_score = predict_risk(
        model,
        time_of_day=features["time_of_day"],
        crowd_density=features["crowd_density"],
        lighting_level=features["lighting_level"],
        crime_history=features["crime_history"],
    )
    time.sleep(0.3)
    steps["Risk Prediction (ML)"] = "done"

    # -- Routing ------------------------------------------------------------------
    steps["Route Optimization (Dijkstra)"] = "active"
    update_pipeline(steps)

    risk_overrides = {
        "A-B": risk_score * 0.4,
        "B-D": risk_score * 0.8,
        "C-F": max(0.5, risk_score * 0.15),
        "E-G": max(0.5, risk_score * 0.2),
    }
    graph = create_sample_graph(risk_scores=risk_overrides)
    route = graph.find_safest_route("A", "G")
    time.sleep(0.3)
    steps["Route Optimization (Dijkstra)"] = "done"
    update_pipeline(steps)

    # =========================================================================
    # RESULTS
    # =========================================================================
    st.markdown("---")

    safety_label, safety_css, safety_color = get_safety_status(risk_score)

    # -- Top metrics row -------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(render_metric("Crowd Density", crowd_density_label.upper()), unsafe_allow_html=True)
    with col2:
        st.markdown(render_metric("People Detected", str(person_count)), unsafe_allow_html=True)
    with col3:
        st.markdown(render_metric("Risk Score", f"{risk_score}/10"), unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="{safety_css}">{safety_label}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -- Two-column layout: video + map ----------------------------------------
    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("📹 Crowd Detection")
        if detected_frame is not None:
            display_frame = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
            st.image(display_frame, caption=f"Detected {person_count} people — density: {crowd_density_label}", use_container_width=True)
        else:
            # Show a placeholder visualization for simulated mode
            fig_data = {
                "Density Level": [crowd_density_label.upper()],
                "Simulated People": [person_count],
            }
            st.info(f"Simulated mode — crowd density: **{crowd_density_label}** ({person_count} people)")
            # Visual bar
            density_pct = {"low": 20, "medium": 55, "high": 90}[crowd_density_label]
            density_color = {"low": "#4caf50", "medium": "#ff9800", "high": "#f44336"}[crowd_density_label]
            st.markdown(
                f'<div style="background:#222;border-radius:8px;padding:4px;">'
                f'<div style="width:{density_pct}%;background:{density_color};'
                f'height:30px;border-radius:6px;text-align:center;line-height:30px;'
                f'color:white;font-weight:bold;">{crowd_density_label.upper()}</div></div>',
                unsafe_allow_html=True,
            )

    with right_col:
        st.subheader("📍 Location Safety Map")
        # Build map dataframe with all locations and color by safety
        map_rows = []
        for loc_key, loc_info in LOCATIONS.items():
            crime = get_historical_crime_score(loc_key)
            loc_features = extract_features(crowd_density=crowd_density_num, location=loc_key, timestamp=timestamp)
            loc_risk = predict_risk(model, **loc_features)
            map_rows.append({
                "lat": loc_info["lat"] + np.random.uniform(-0.002, 0.002),
                "lon": loc_info["lon"] + np.random.uniform(-0.002, 0.002),
                "name": loc_info["name"],
                "risk": loc_risk,
                "size": 300 + loc_risk * 80,
                "color_r": int(min(255, loc_risk * 28)),
                "color_g": int(max(0, 255 - loc_risk * 28)),
                "color_b": 50,
                "color_a": 180,
            })

        map_df = pd.DataFrame(map_rows)

        st.pydeck_chart(
            pdk.Deck(
                initial_view_state=pdk.ViewState(
                    latitude=location_info["lat"],
                    longitude=location_info["lon"],
                    zoom=13,
                    pitch=40,
                ),
                layers=[
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=map_df,
                        get_position=["lon", "lat"],
                        get_radius="size",
                        get_fill_color=["color_r", "color_g", "color_b", "color_a"],
                        pickable=True,
                    ),
                ],
                tooltip={"text": "{name}\nRisk: {risk}/10"},
            )
        )

        st.caption(f"Current location: **{location_info['name']}** — Risk: **{risk_score}/10**")

    # -- Features & Route details -----------------------------------------------
    st.markdown("---")
    detail_left, detail_right = st.columns(2)

    with detail_left:
        st.subheader("🔬 Extracted Features")
        feat_df = pd.DataFrame([features])
        feat_df.columns = ["Time of Day", "Crowd Density", "Lighting Level", "Crime History"]
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

        lighting_labels = {0: "Dark 🌑", 1: "Dim 🌗", 2: "Well-lit ☀️"}
        st.markdown(f"**Lighting:** {lighting_labels.get(features['lighting_level'], 'Unknown')}")

    with detail_right:
        st.subheader("🗺️ Recommended Safest Route")
        if route:
            route_str = " → ".join(route.path)
            st.success(f"**{route_str}**")
            st.markdown(f"- **Total Distance:** {route.total_distance}")
            st.markdown(f"- **Total Risk:** {route.total_risk}")
            st.markdown(f"- **Total Cost:** {route.total_cost}")

            seg_data = []
            for seg in route.segment_details:
                seg_data.append({
                    "Segment": f"{seg['from']} → {seg['to']}",
                    "Distance": seg["distance"],
                    "Risk": round(seg["risk_score"], 1),
                    "Cost": round(seg["cost"], 1),
                })
            st.dataframe(pd.DataFrame(seg_data), use_container_width=True, hide_index=True)
        else:
            st.error("No route found!")

    # -- Risk gauge ----------------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Risk Score Gauge")

    gauge_pct = risk_score * 10
    gauge_color = safety_color
    st.markdown(
        f'<div style="background:#1e1e2f;border-radius:12px;padding:10px 20px;">'
        f'<div style="display:flex;align-items:center;gap:12px;">'
        f'<span style="color:white;font-size:18px;min-width:40px;">{risk_score}</span>'
        f'<div style="flex:1;background:#333;border-radius:8px;height:28px;">'
        f'<div style="width:{gauge_pct}%;background:{gauge_color};height:28px;'
        f'border-radius:8px;transition:width 0.5s;"></div></div>'
        f'<span style="color:white;font-size:18px;">10</span>'
        f'</div>'
        f'<div style="display:flex;justify-content:space-between;margin-top:6px;">'
        f'<span style="color:#4caf50;font-size:12px;">Safe</span>'
        f'<span style="color:#ff9800;font-size:12px;">Moderate</span>'
        f'<span style="color:#f44336;font-size:12px;">High Risk</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # -- Pipeline summary ----------------------------------------------------------
    st.markdown("---")
    st.markdown(
        f'<div style="background:#1a1a2e;padding:20px;border-radius:12px;text-align:center;">'
        f'<span style="color:#aaa;font-size:13px;">PIPELINE RESULT</span><br>'
        f'<span style="color:white;font-size:20px;">'
        f'📹 Video → 👥 {crowd_density_label.upper()} → 🔬 Features → '
        f'🤖 Risk: <span style="color:{safety_color};font-weight:bold;">{risk_score}/10</span> → '
        f'🗺️ Route: <span style="color:#64b5f6;">{" → ".join(route.path) if route else "N/A"}</span>'
        f'</span></div>',
        unsafe_allow_html=True,
    )

else:
    # Landing state
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;padding:60px 20px;color:#888;">'
        '<p style="font-size:48px;">🛡️</p>'
        '<h2>Welcome to SafeRoute AI</h2>'
        '<p>Configure parameters in the sidebar, then click '
        '<strong>Run SafeRoute AI Pipeline</strong> to analyze safety.</p>'
        '<p style="font-size:14px;margin-top:20px;opacity:0.6;">'
        'video → crowd detection → feature extraction → risk model → safety score → routing</p>'
        '</div>',
        unsafe_allow_html=True,
    )
