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

## Visual Dashboard (Streamlit)

Launch the interactive UI:

```bash
streamlit run app.py
```

The dashboard features:
- **Crowd density** detection with bounding boxes on video frames
- **Risk score** with color-coded safety status (green/yellow/red)
- **Location safety map** showing risk levels across road segments
- **Safest route** recommendation via Dijkstra's algorithm
- **Step-by-step pipeline progress** visualization

Use the sidebar to select a road segment, simulate time of day and crowd density, or toggle live YOLOv8 crowd detection from the sample video.

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
  /api                   # API module (extensible)
  app.py                 # Streamlit visual dashboard
  main.py                # Full pipeline demo (console)
  requirements.txt       # Python dependencies
```
