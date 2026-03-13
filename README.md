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
  main.py                # Full pipeline demo
  requirements.txt       # Python dependencies
```
