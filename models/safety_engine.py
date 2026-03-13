"""
Module 4 — Safety Engine.

Combines crowd detection output with the ML risk model to produce
a safety score for road segments.

Pipeline: video → crowd detection → feature extraction → risk model → safety score
"""

from datetime import datetime

import numpy as np

from vision.crowd_detector import CrowdDetector
from models.feature_extractor import extract_features
from models.risk_model import load_model, predict_risk


class SafetyEngine:
    """
    Main safety analysis engine that combines all modules.

    Pipeline:
        video → crowd detection → feature extraction → risk model → safety score
    """

    def __init__(self, model_path: str | None = None, yolo_model: str = "yolov8n.pt"):
        """
        Initialize the safety engine.

        Args:
            model_path: Path to trained risk model. If None, loads default.
            yolo_model: Path to YOLOv8 model weights.
        """
        self.crowd_detector = CrowdDetector(model_path=yolo_model)
        self.risk_model = load_model(model_path)

    def analyze_frame(
        self,
        frame: np.ndarray,
        location: str = "road_segment_1",
        timestamp: datetime | None = None,
        crime_data: dict | None = None,
    ) -> dict:
        """
        Analyze a single frame and return a safety assessment.

        Args:
            frame: BGR image as numpy array.
            location: Road segment identifier.
            timestamp: Current time (defaults to now).
            crime_data: Optional crime score mapping.

        Returns:
            Safety assessment dictionary.
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Step 1: Crowd detection
        crowd_info = self.crowd_detector.get_crowd_density(frame)

        # Step 2: Feature extraction
        features = extract_features(
            crowd_density=crowd_info["density_numeric"],
            location=location,
            timestamp=timestamp,
            crime_data=crime_data,
        )

        # Step 3: Risk prediction
        risk_score = predict_risk(
            self.risk_model,
            time_of_day=features["time_of_day"],
            crowd_density=features["crowd_density"],
            lighting_level=features["lighting_level"],
            crime_history=features["crime_history"],
        )

        return {
            "location": location,
            "time": timestamp.strftime("%H:%M"),
            "crowd_density": crowd_info["density"],
            "person_count": crowd_info["person_count"],
            "features": features,
            "risk_score": risk_score,
        }

    def analyze_video(
        self,
        video_path: str,
        location: str = "road_segment_1",
        sample_rate: int = 30,
        timestamp: datetime | None = None,
        crime_data: dict | None = None,
    ) -> dict:
        """
        Analyze a video file and return aggregate safety assessment.

        Args:
            video_path: Path to the video file.
            location: Road segment identifier.
            sample_rate: Process every Nth frame.
            timestamp: Override timestamp (defaults to now).
            crime_data: Optional crime score mapping.

        Returns:
            Aggregate safety assessment with per-frame details.
        """
        import cv2

        if timestamp is None:
            timestamp = datetime.now()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frame_results = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                result = self.analyze_frame(frame, location, timestamp, crime_data)
                result["frame"] = frame_idx
                frame_results.append(result)

            frame_idx += 1

        cap.release()

        if not frame_results:
            return {
                "location": location,
                "time": timestamp.strftime("%H:%M"),
                "crowd_density": "low",
                "person_count": 0,
                "risk_score": 0.0,
                "frames_analyzed": 0,
            }

        # Aggregate results
        avg_risk = round(
            sum(r["risk_score"] for r in frame_results) / len(frame_results), 1
        )
        avg_person_count = sum(r["person_count"] for r in frame_results) / len(frame_results)

        # Determine overall crowd density from average person count
        from vision.crowd_detector import classify_density
        overall_density = classify_density(int(avg_person_count))

        return {
            "location": location,
            "time": timestamp.strftime("%H:%M"),
            "crowd_density": overall_density,
            "avg_person_count": round(avg_person_count, 1),
            "risk_score": avg_risk,
            "frames_analyzed": len(frame_results),
            "frame_details": frame_results,
        }

    def analyze_location(
        self,
        crowd_density_numeric: int,
        location: str = "road_segment_1",
        timestamp: datetime | None = None,
        crime_data: dict | None = None,
    ) -> dict:
        """
        Analyze a location without video input (using pre-computed crowd density).

        Useful when crowd density is already known or estimated.

        Args:
            crowd_density_numeric: 0=low, 1=medium, 2=high.
            location: Road segment identifier.
            timestamp: Override timestamp (defaults to now).
            crime_data: Optional crime score mapping.

        Returns:
            Safety assessment dictionary.
        """
        if timestamp is None:
            timestamp = datetime.now()

        from vision.crowd_detector import classify_density
        density_labels = {0: "low", 1: "medium", 2: "high"}

        features = extract_features(
            crowd_density=crowd_density_numeric,
            location=location,
            timestamp=timestamp,
            crime_data=crime_data,
        )

        risk_score = predict_risk(
            self.risk_model,
            time_of_day=features["time_of_day"],
            crowd_density=features["crowd_density"],
            lighting_level=features["lighting_level"],
            crime_history=features["crime_history"],
        )

        return {
            "location": location,
            "time": timestamp.strftime("%H:%M"),
            "crowd_density": density_labels.get(crowd_density_numeric, "low"),
            "features": features,
            "risk_score": risk_score,
        }
