"""
SafeRoute AI — Urban Safety Dashboard (Delhi)

Interactive Streamlit dashboard with real Delhi locations,
safety heatmap, route scenarios, and the full AI pipeline.

This is a page in the multi-page app. Entry point is landing.py.
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vision.crowd_detector import CrowdDetector, classify_density, density_to_numeric
from models.feature_extractor import extract_features, estimate_lighting_level
from models.risk_model import load_model, predict_risk
from routing.dijkstra import SafeRouteGraph

# ---------------------------------------------------------------------------
# Page config — in multi-page apps only the entry point (app.py) calls
# set_page_config.  This page inherits its settings automatically.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Delhi locations with simulated safety data
# ---------------------------------------------------------------------------
DELHI_LOCATIONS = {
    "Kalkaji": {
        "lat": 28.5355, "lon": 77.2588,
        "crime_history": 2, "base_crowd": "medium",
        "description": "Residential area near Kalkaji Mandir",
    },
    "Nehru Place": {
        "lat": 28.5491, "lon": 77.2533,
        "crime_history": 3, "base_crowd": "high",
        "description": "Major commercial IT hub",
    },
    "Govindpuri": {
        "lat": 28.5390, "lon": 77.2640,
        "crime_history": 4, "base_crowd": "medium",
        "description": "Dense residential colony",
    },
    "Lajpat Nagar": {
        "lat": 28.5700, "lon": 77.2400,
        "crime_history": 2, "base_crowd": "high",
        "description": "Popular market and shopping area",
    },
    "Greater Kailash": {
        "lat": 28.5494, "lon": 77.2340,
        "crime_history": 1, "base_crowd": "low",
        "description": "Upscale residential neighbourhood",
    },
    "Saket": {
        "lat": 28.5244, "lon": 77.2167,
        "crime_history": 1, "base_crowd": "medium",
        "description": "Malls, dining, and residential",
    },
    "Hauz Khas": {
        "lat": 28.5539, "lon": 77.2060,
        "crime_history": 2, "base_crowd": "medium",
        "description": "Cultural hub with cafes and nightlife",
    },
    "Okhla": {
        "lat": 28.5310, "lon": 77.2710,
        "crime_history": 3, "base_crowd": "medium",
        "description": "Industrial and residential zone",
    },
    "AIIMS": {
        "lat": 28.5672, "lon": 77.2100,
        "crime_history": 1, "base_crowd": "high",
        "description": "Hospital area, Green Park vicinity",
    },
}

# Predefined route scenarios
ROUTE_SCENARIOS = {
    "Kalkaji → Nehru Place": ("Kalkaji", "Nehru Place"),
    "Lajpat Nagar → Greater Kailash": ("Lajpat Nagar", "Greater Kailash"),
    "Saket → Hauz Khas": ("Saket", "Hauz Khas"),
    "Custom Route": None,
}


def build_delhi_graph(location_risks: dict[str, float]) -> SafeRouteGraph:
    """
    Build a road-network graph connecting Delhi locations.

    Distances are approximate straight-line km between places.
    Risk scores come from the ML model predictions.
    """
    graph = SafeRouteGraph()

    # (from, to, approx distance in km)
    connections = [
        ("Kalkaji", "Nehru Place", 1.8),
        ("Kalkaji", "Govindpuri", 1.2),
        ("Kalkaji", "Okhla", 2.0),
        ("Nehru Place", "Greater Kailash", 2.5),
        ("Nehru Place", "Govindpuri", 2.0),
        ("Govindpuri", "Okhla", 1.5),
        ("Lajpat Nagar", "Greater Kailash", 2.0),
        ("Lajpat Nagar", "Nehru Place", 2.8),
        ("Lajpat Nagar", "AIIMS", 3.2),
        ("Greater Kailash", "Saket", 3.0),
        ("Greater Kailash", "Hauz Khas", 3.5),
        ("Saket", "Hauz Khas", 3.8),
        ("Saket", "Okhla", 4.0),
        ("Hauz Khas", "AIIMS", 2.2),
        ("AIIMS", "Lajpat Nagar", 3.2),
        ("AIIMS", "Saket", 3.5),
    ]

    for src, dst, dist in connections:
        avg_risk = (location_risks.get(src, 5) + location_risks.get(dst, 5)) / 2
        graph.add_edge(src, dst, dist, round(avg_risk, 1))

    return graph


def get_safety_status(risk_score: float) -> tuple[str, str, str]:
    """Return (label, css_class, color) based on risk score."""
    if risk_score <= 3.5:
        return "SAFE", "safe-box", "#00c853"
    elif risk_score <= 6.5:
        return "MODERATE RISK", "moderate-box", "#ff9800"
    else:
        return "HIGH RISK", "danger-box", "#f44336"


def safety_color_rgb(risk: float) -> list[int]:
    """Return [R, G, B, A] for a risk value 0-10."""
    r = int(min(255, risk * 28))
    g = int(max(0, 255 - risk * 28))
    return [r, g, 60, 200]


def render_metric(label: str, value: str) -> str:
    return f'<div class="metric-card"><h3>{label}</h3><h1>{value}</h1></div>'


def pipeline_step_html(text: str, state: str = "pending") -> str:
    icons = {"pending": "⏳", "active": "🔄", "done": "✅"}
    return f'<div class="pipeline-step {state}">{icons.get(state, "⏳")}  {text}</div>'


def draw_bounding_boxes(frame: np.ndarray, boxes: list[dict], density: str) -> np.ndarray:
    color_map = {"low": (0, 200, 0), "medium": (0, 200, 255), "high": (0, 0, 255)}
    color = color_map.get(density, (0, 200, 0))
    annotated = frame.copy()
    for box in boxes:
        cv2.rectangle(annotated, (box["x1"], box["y1"]), (box["x2"], box["y2"]), color, 2)
        label = f"person {box['confidence']:.0%}"
        cv2.putText(annotated, label, (box["x1"], box["y1"] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return annotated


# ---------------------------------------------------------------------------
# Custom CSS
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
    .loc-table { width:100%; border-collapse:collapse; }
    .loc-table th {
        background:#1a1a2e; color:#90a4ae; padding:10px 12px;
        text-align:left; font-size:13px; border-bottom:1px solid #2a2a4a;
    }
    .loc-table td {
        padding:10px 12px; border-bottom:1px solid #1e1e38;
        font-size:14px; color:#ccc;
    }
    .badge {
        display:inline-block; padding:3px 10px; border-radius:20px;
        font-size:12px; font-weight:600; color:#fff;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/route.png", width=64)
st.sidebar.title("SafeRoute AI")
st.sidebar.markdown("---")

sim_hour = st.sidebar.slider("Simulate Hour of Day", 0, 23, 22)
sim_crowd = st.sidebar.selectbox("Override Crowd Density", ["auto", "low", "medium", "high"])

st.sidebar.markdown("#### Route Scenario")
scenario_key = st.sidebar.selectbox("Choose route", list(ROUTE_SCENARIOS.keys()))

if scenario_key == "Custom Route":
    loc_names = list(DELHI_LOCATIONS.keys())
    route_start = st.sidebar.selectbox("From", loc_names, index=0)
    route_end = st.sidebar.selectbox("To", loc_names, index=1)
else:
    route_start, route_end = ROUTE_SCENARIOS[scenario_key]

use_video = st.sidebar.checkbox("Run YOLOv8 Crowd Detection", value=False)

st.sidebar.markdown("---")
st.sidebar.caption("SafeRoute AI v2.0 — Delhi Demo")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🛡️ SafeRoute AI — Delhi Safety Dashboard")
st.caption("Real-time AI pipeline: video → crowd detection → risk prediction → safe routing")

pipeline_placeholder = st.empty()


def update_pipeline(steps: dict[str, str]) -> None:
    html = '<div style="margin-bottom:18px;">'
    for text, state in steps.items():
        html += pipeline_step_html(text, state)
    html += "</div>"
    pipeline_placeholder.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
if st.button("▶  Run SafeRoute AI Pipeline", type="primary", use_container_width=True):

    steps = {
        "Initializing model...": "active",
        "Crowd Detection (YOLOv8)": "pending",
        "Feature Extraction — all Delhi locations": "pending",
        "Risk Prediction (ML)": "pending",
        "Route Optimization (Dijkstra)": "pending",
    }
    update_pipeline(steps)
    time.sleep(0.3)

    model = load_model()
    steps["Initializing model..."] = "done"

    # -- Crowd detection --------------------------------------------------
    steps["Crowd Detection (YOLOv8)"] = "active"
    update_pipeline(steps)

    detected_frame = None
    video_crowd_label = None

    if use_video:
        video_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_video.avi")
        if not os.path.exists(video_path):
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from data.generate_sample_video import generate_sample_video
            generate_sample_video(output_path=video_path)

        detector = CrowdDetector(model_path="yolov8n.pt", confidence=0.4)
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            crowd_info = detector.get_crowd_density(frame)
            video_crowd_label = crowd_info["density"]
            detected_frame = draw_bounding_boxes(frame, crowd_info["boxes"], video_crowd_label)

    time.sleep(0.3)
    steps["Crowd Detection (YOLOv8)"] = "done"

    # -- Feature extraction & risk for every location -----------------------
    steps["Feature Extraction — all Delhi locations"] = "active"
    update_pipeline(steps)

    timestamp = datetime(2026, 3, 13, sim_hour, 30)
    lighting = estimate_lighting_level(sim_hour)
    lighting_labels = {0: "Dark 🌑", 1: "Dim 🌗", 2: "Well-lit ☀️"}

    location_data: dict[str, dict] = {}
    for loc_name, info in DELHI_LOCATIONS.items():
        # Determine crowd density for this location
        if sim_crowd != "auto":
            crowd_label = sim_crowd
        elif video_crowd_label and loc_name == route_start:
            crowd_label = video_crowd_label
        else:
            crowd_label = info["base_crowd"]

        crowd_num = density_to_numeric(crowd_label)
        crime = info["crime_history"]

        features = {
            "time_of_day": sim_hour,
            "crowd_density": crowd_num,
            "lighting_level": lighting,
            "crime_history": crime,
        }
        risk = predict_risk(model, **features)
        status_label, status_css, status_color = get_safety_status(risk)

        location_data[loc_name] = {
            **info,
            "crowd_density": crowd_label,
            "crowd_num": crowd_num,
            "lighting_level": lighting,
            "features": features,
            "risk_score": risk,
            "safety_label": status_label,
            "safety_color": status_color,
        }

    time.sleep(0.2)
    steps["Feature Extraction — all Delhi locations"] = "done"
    steps["Risk Prediction (ML)"] = "active"
    update_pipeline(steps)
    time.sleep(0.3)
    steps["Risk Prediction (ML)"] = "done"

    # -- Routing -----------------------------------------------------------
    steps["Route Optimization (Dijkstra)"] = "active"
    update_pipeline(steps)

    risk_map = {name: d["risk_score"] for name, d in location_data.items()}
    graph = build_delhi_graph(risk_map)
    route = graph.find_safest_route(route_start, route_end)

    time.sleep(0.3)
    steps["Route Optimization (Dijkstra)"] = "done"
    update_pipeline(steps)

    # =====================================================================
    # RESULTS
    # =====================================================================
    st.markdown("---")

    start_risk = location_data[route_start]["risk_score"]
    safety_label, safety_css, safety_color = get_safety_status(start_risk)

    start_crowd = location_data[route_start]["crowd_density"]
    start_people = {"low": 2, "medium": 10, "high": 22}[start_crowd]

    # -- Top metrics -------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(render_metric("Crowd Density", start_crowd.upper()), unsafe_allow_html=True)
    with c2:
        st.markdown(render_metric("People Detected", str(start_people)), unsafe_allow_html=True)
    with c3:
        st.markdown(render_metric("Risk Score", f"{start_risk}/10"), unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="{safety_css}">{safety_label}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -- Map + crowd detection columns -------------------------------------
    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("📹 Crowd Detection")
        if detected_frame is not None:
            display = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
            st.image(display, caption=f"YOLOv8 detection — density: {video_crowd_label}", use_container_width=True)
        else:
            density_pct = {"low": 20, "medium": 55, "high": 90}[start_crowd]
            density_color = {"low": "#4caf50", "medium": "#ff9800", "high": "#f44336"}[start_crowd]
            st.info(f"Simulated — **{route_start}** crowd density: **{start_crowd}** ({start_people} people)")
            st.markdown(
                f'<div style="background:#222;border-radius:8px;padding:4px;">'
                f'<div style="width:{density_pct}%;background:{density_color};'
                f'height:30px;border-radius:6px;text-align:center;line-height:30px;'
                f'color:white;font-weight:bold;">{start_crowd.upper()}</div></div>',
                unsafe_allow_html=True,
            )

    with right_col:
        st.subheader("📍 Delhi Safety Map")

        # Build marker data
        markers = []
        for name, d in location_data.items():
            c = safety_color_rgb(d["risk_score"])
            markers.append({
                "name": name,
                "lat": d["lat"],
                "lon": d["lon"],
                "risk": d["risk_score"],
                "status": d["safety_label"],
                "size": 250 + d["risk_score"] * 60,
                "color_r": c[0], "color_g": c[1], "color_b": c[2], "color_a": c[3],
            })

        marker_df = pd.DataFrame(markers)

        # Build route line layer
        layers = [
            pdk.Layer(
                "ScatterplotLayer",
                data=marker_df,
                get_position=["lon", "lat"],
                get_radius="size",
                get_fill_color=["color_r", "color_g", "color_b", "color_a"],
                pickable=True,
            ),
        ]

        # Add route path as an ArcLayer if route was found
        if route:
            arc_data = []
            for seg in route.segment_details:
                src = location_data[seg["from"]]
                dst = location_data[seg["to"]]
                arc_data.append({
                    "from_lat": src["lat"], "from_lon": src["lon"],
                    "to_lat": dst["lat"], "to_lon": dst["lon"],
                    "seg_name": f"{seg['from']} → {seg['to']}",
                })
            arc_df = pd.DataFrame(arc_data)
            layers.append(
                pdk.Layer(
                    "ArcLayer",
                    data=arc_df,
                    get_source_position=["from_lon", "from_lat"],
                    get_target_position=["to_lon", "to_lat"],
                    get_source_color=[0, 180, 255, 220],
                    get_target_color=[0, 255, 130, 220],
                    get_width=4,
                    pickable=True,
                ),
            )

        center_lat = sum(d["lat"] for d in DELHI_LOCATIONS.values()) / len(DELHI_LOCATIONS)
        center_lon = sum(d["lon"] for d in DELHI_LOCATIONS.values()) / len(DELHI_LOCATIONS)

        st.pydeck_chart(
            pdk.Deck(
                initial_view_state=pdk.ViewState(
                    latitude=center_lat, longitude=center_lon,
                    zoom=12.3, pitch=35,
                ),
                layers=layers,
                tooltip={"text": "{name}\nRisk: {risk}/10\nStatus: {status}"},
            )
        )

    # -- Location safety table ---------------------------------------------
    st.markdown("---")
    st.subheader("📊 All Delhi Locations — Safety Overview")

    table_html = '<table class="loc-table"><tr><th>Location</th><th>Description</th><th>Crowd</th><th>Crime</th><th>Risk</th><th>Status</th></tr>'
    for name, d in location_data.items():
        badge_bg = d["safety_color"]
        table_html += (
            f'<tr><td><strong>{name}</strong></td>'
            f'<td>{d["description"]}</td>'
            f'<td>{d["crowd_density"].capitalize()}</td>'
            f'<td>{d["crime_history"]}/5</td>'
            f'<td>{d["risk_score"]}/10</td>'
            f'<td><span class="badge" style="background:{badge_bg}">{d["safety_label"]}</span></td>'
            f'</tr>'
        )
    table_html += '</table>'
    st.markdown(table_html, unsafe_allow_html=True)

    # -- Features & Route --------------------------------------------------
    st.markdown("---")
    dl, dr = st.columns(2)

    with dl:
        st.subheader("🔬 Features — " + route_start)
        feat_df = pd.DataFrame([location_data[route_start]["features"]])
        feat_df.columns = ["Time of Day", "Crowd Density", "Lighting Level", "Crime History"]
        st.dataframe(feat_df, use_container_width=True, hide_index=True)
        st.markdown(f"**Lighting:** {lighting_labels.get(lighting, 'Unknown')}")

    with dr:
        st.subheader(f"🗺️ Safest Route: {route_start} → {route_end}")
        if route:
            st.success(f"**{' → '.join(route.path)}**")
            st.markdown(f"- **Total Distance:** {route.total_distance} km")
            st.markdown(f"- **Total Risk:** {route.total_risk}")
            st.markdown(f"- **Total Cost:** {route.total_cost}")
            seg_rows = [{
                "Segment": f"{s['from']} → {s['to']}",
                "Distance (km)": s["distance"],
                "Risk": round(s["risk_score"], 1),
                "Cost": round(s["cost"], 1),
            } for s in route.segment_details]
            st.dataframe(pd.DataFrame(seg_rows), use_container_width=True, hide_index=True)
        else:
            st.error("No route found between these locations!")

    # -- Risk gauge --------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Risk Score Gauge")
    gauge_pct = start_risk * 10
    st.markdown(
        f'<div style="background:#1e1e2f;border-radius:12px;padding:10px 20px;">'
        f'<div style="display:flex;align-items:center;gap:12px;">'
        f'<span style="color:white;font-size:18px;min-width:40px;">{start_risk}</span>'
        f'<div style="flex:1;background:#333;border-radius:8px;height:28px;">'
        f'<div style="width:{gauge_pct}%;background:{safety_color};height:28px;'
        f'border-radius:8px;"></div></div>'
        f'<span style="color:white;font-size:18px;">10</span></div>'
        f'<div style="display:flex;justify-content:space-between;margin-top:6px;">'
        f'<span style="color:#4caf50;font-size:12px;">Safe</span>'
        f'<span style="color:#ff9800;font-size:12px;">Moderate</span>'
        f'<span style="color:#f44336;font-size:12px;">High Risk</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # -- Pipeline summary banner -------------------------------------------
    st.markdown("---")
    route_path_str = " → ".join(route.path) if route else "N/A"
    st.markdown(
        f'<div style="background:#1a1a2e;padding:20px;border-radius:12px;text-align:center;">'
        f'<span style="color:#aaa;font-size:13px;">PIPELINE RESULT</span><br>'
        f'<span style="color:white;font-size:20px;">'
        f'📹 Video → 👥 {start_crowd.upper()} → 🔬 Features → '
        f'🤖 Risk: <span style="color:{safety_color};font-weight:bold;">{start_risk}/10</span> → '
        f'🗺️ Route: <span style="color:#64b5f6;">{route_path_str}</span>'
        f'</span></div>',
        unsafe_allow_html=True,
    )

else:
    # Landing state
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;padding:60px 20px;color:#888;">'
        '<p style="font-size:48px;">🛡️</p>'
        '<h2>Welcome to SafeRoute AI — Delhi</h2>'
        '<p>Configure parameters in the sidebar, choose a route scenario,<br>'
        'then click <strong>Run SafeRoute AI Pipeline</strong>.</p>'
        '<p style="font-size:14px;margin-top:20px;opacity:0.6;">'
        'video → crowd detection → feature extraction → risk model → safety score → routing</p>'
        '</div>',
        unsafe_allow_html=True,
    )
