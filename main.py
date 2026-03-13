"""
SafeRoute AI — Main Pipeline Script

Demonstrates the full system pipeline:
    video → crowd detection → feature extraction → risk model → safety score → routing

Prints:
    - Crowd density
    - Predicted risk score
    - Recommended safest route
"""

import os
import sys
from datetime import datetime

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from vision.crowd_detector import CrowdDetector, classify_density, density_to_numeric
from models.feature_extractor import extract_features
from models.risk_model import (
    generate_sample_dataset,
    train_model,
    load_model,
    predict_risk,
)
from models.safety_engine import SafetyEngine
from routing.dijkstra import SafeRouteGraph, create_sample_graph


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def step1_generate_data() -> None:
    """Step 1: Generate sample dataset for model training."""
    print_header("Step 1: Generating Sample Dataset")
    df = generate_sample_dataset(n_samples=2000, save=True)
    print(f"Dataset shape: {df.shape}")
    print(f"Risk score distribution:\n{df['risk_score'].value_counts().sort_index()}")


def step2_train_model():
    """Step 2: Train the risk prediction model."""
    print_header("Step 2: Training Risk Prediction Model")
    model = train_model()
    return model


def step3_crowd_detection(video_path: str) -> dict:
    """Step 3: Run crowd detection on sample video."""
    print_header("Step 3: Crowd Detection (YOLOv8)")

    detector = CrowdDetector(model_path="yolov8n.pt", confidence=0.4)

    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Cannot open video {video_path}")
        print("Using simulated crowd density instead.")
        return {"person_count": 3, "density": "low", "density_numeric": 0}

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Warning: Cannot read frame from video.")
        return {"person_count": 3, "density": "low", "density_numeric": 0}

    result = detector.get_crowd_density(frame)
    print(f"People detected: {result['person_count']}")
    print(f"Crowd density: {result['density']}")
    return result


def step4_feature_extraction(crowd_density_numeric: int, location: str, timestamp: datetime) -> dict:
    """Step 4: Extract features for the risk model."""
    print_header("Step 4: Feature Extraction")

    features = extract_features(
        crowd_density=crowd_density_numeric,
        location=location,
        timestamp=timestamp,
    )

    print("Extracted features:")
    for key, value in features.items():
        print(f"  {key}: {value}")

    return features


def step5_risk_prediction(model, features: dict) -> float:
    """Step 5: Predict risk score."""
    print_header("Step 5: Risk Prediction")

    risk_score = predict_risk(
        model,
        time_of_day=features["time_of_day"],
        crowd_density=features["crowd_density"],
        lighting_level=features["lighting_level"],
        crime_history=features["crime_history"],
    )

    print(f"Predicted Risk Score: {risk_score}/10")
    return risk_score


def step6_routing(risk_score: float) -> None:
    """Step 6: Find safest route using Dijkstra's algorithm."""
    print_header("Step 6: Safest Route (Dijkstra's Algorithm)")

    # Create graph with varying risk scores
    # Use the predicted risk_score to influence some edges
    risk_overrides = {
        "A-B": risk_score * 0.4,
        "B-D": risk_score * 0.8,
        "C-F": max(0.5, risk_score * 0.15),
        "E-G": max(0.5, risk_score * 0.2),
    }

    graph = create_sample_graph(risk_scores=risk_overrides)

    start, end = "A", "G"
    result = graph.find_safest_route(start, end)

    if result:
        print(f"From: {start} → To: {end}")
        print(f"\nRecommended Safest Route: {' → '.join(result.path)}")
        print(f"  Total Distance: {result.total_distance}")
        print(f"  Total Risk: {result.total_risk}")
        print(f"  Total Cost (distance + risk): {result.total_cost}")
        print(f"\nRoute Segments:")
        for seg in result.segment_details:
            print(
                f"  {seg['from']} → {seg['to']}: "
                f"distance={seg['distance']:.1f}, "
                f"risk={seg['risk_score']:.1f}, "
                f"cost={seg['cost']:.1f}"
            )
    else:
        print("No route found!")


def run_full_pipeline() -> None:
    """Run the complete SafeRoute AI pipeline."""
    print("\n" + "▓" * 60)
    print("  SafeRoute AI — Full Pipeline Demo")
    print("▓" * 60)

    # Configuration
    location = "road_segment_12"
    timestamp = datetime(2026, 3, 13, 23, 30)  # Late night scenario
    video_path = os.path.join(os.path.dirname(__file__), "data", "sample_video.avi")

    # Generate sample video if it doesn't exist
    if not os.path.exists(video_path):
        print("\nGenerating sample test video...")
        from data.generate_sample_video import generate_sample_video
        generate_sample_video(output_path=video_path)

    # Step 1: Generate training data
    step1_generate_data()

    # Step 2: Train model
    model = step2_train_model()

    # Step 3: Crowd detection
    crowd_info = step3_crowd_detection(video_path)

    # Step 4: Feature extraction
    features = step4_feature_extraction(
        crowd_density_numeric=crowd_info["density_numeric"],
        location=location,
        timestamp=timestamp,
    )

    # Step 5: Risk prediction
    risk_score = step5_risk_prediction(model, features)

    # Step 6: Routing
    step6_routing(risk_score)

    # Final summary
    print_header("Pipeline Summary")
    print(f"  Location:       {location}")
    print(f"  Time:           {timestamp.strftime('%H:%M')}")
    print(f"  Crowd Density:  {crowd_info['density']}")
    print(f"  Risk Score:     {risk_score}/10")
    print(f"\n{'='*60}")
    print("  SafeRoute AI pipeline completed successfully!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_full_pipeline()
