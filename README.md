# SafeRoute AI

An AI pipeline that estimates urban safety risk for road segments and outputs a safety score for navigation.

## Modules

1. **Crowd Detection** (`vision/crowd_detector.py`) — YOLOv8 + OpenCV people detection with crowd density classification
2. **Feature Extraction** (`models/feature_extractor.py`) — Converts environmental signals into ML features
3. **Risk Prediction** (`models/risk_model.py`) — RandomForestClassifier for safety risk scoring (0–10)
4. **Safety Engine** (`models/safety_engine.py`) — Combines all modules into one pipeline
5. **Routing Logic** (`routing/dijkstra.py`) — Dijkstra's algorithm with cost = distance + risk_score

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

## Visual Demo (Streamlit)

Launch the app:

```bash
streamlit run app.py
```

This opens the **landing page** with a hero section and feature overview.
Click **"Open Safety Map"** to enter the interactive **Delhi Safety Dashboard**.

### Landing Page

- Project title and description
- Feature cards: AI Risk Prediction, Crowd Detection, Safety Heatmap, Safe Route Navigation
- "How It Works" pipeline flow
- CTA button to launch the dashboard

### Delhi Safety Dashboard

- **9 real Delhi locations**: Kalkaji, Nehru Place, Govindpuri, Lajpat Nagar, Greater Kailash, Saket, Hauz Khas, Okhla, AIIMS / Green Park
- **Color-coded safety markers** on an interactive map (green = safe, yellow = moderate, red = high risk)
- **Sample route scenarios**: Kalkaji → Nehru Place, Lajpat Nagar → Greater Kailash, Saket → Hauz Khas
- **Safest route** highlighted using Dijkstra's algorithm with arc visualization
- **Full AI pipeline progress**: model init → crowd detection → feature extraction → risk prediction → routing
- **Safety overview table** for all locations
- Sidebar controls: time of day, crowd density override, route selection, optional YOLOv8 video detection

## Pipeline

```
video → crowd detection → feature extraction → risk model → safety score → routing
```

## Project Structure

```
/saferoute-ai
  /data                  # Sample dataset & video generator
  /models                # Feature extraction, risk model, safety engine
  /vision                # YOLOv8 crowd detection
  /routing               # Dijkstra routing algorithm
  /pages
    dashboard.py         # Delhi Safety Dashboard (multi-page)
  /api                   # API module (extensible)
  app.py                 # Landing page + entry point (streamlit run app.py)
  landing.py             # Standalone landing page (alternative entry)
  main.py                # Full pipeline demo (console)
  requirements.txt       # Python dependencies
```
