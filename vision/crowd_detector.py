"""
Module 1 — Crowd Detection using YOLOv8 and OpenCV.

Detects people in video frames and outputs crowd density levels.
"""

import cv2
import numpy as np
from ultralytics import YOLO


# Crowd density thresholds
DENSITY_LOW_THRESHOLD = 5
DENSITY_MEDIUM_THRESHOLD = 15


def classify_density(person_count: int) -> str:
    """Classify crowd density based on the number of people detected."""
    if person_count <= DENSITY_LOW_THRESHOLD:
        return "low"
    elif person_count <= DENSITY_MEDIUM_THRESHOLD:
        return "medium"
    else:
        return "high"


def density_to_numeric(density: str) -> int:
    """Convert density label to a numeric value for ML features."""
    mapping = {"low": 0, "medium": 1, "high": 2}
    return mapping.get(density, 0)


class CrowdDetector:
    """Detects people in video frames using YOLOv8."""

    # COCO class ID for 'person'
    PERSON_CLASS_ID = 0

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.4):
        """
        Initialize the crowd detector.

        Args:
            model_path: Path to YOLOv8 model weights.
            confidence: Minimum confidence threshold for detections.
        """
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect_people(self, frame: np.ndarray) -> tuple[int, list]:
        """
        Detect people in a single frame.

        Args:
            frame: BGR image as numpy array.

        Returns:
            Tuple of (person_count, list of bounding boxes).
        """
        results = self.model(frame, conf=self.confidence, verbose=False)
        boxes = []
        person_count = 0

        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == self.PERSON_CLASS_ID:
                    person_count += 1
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    boxes.append({
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                        "confidence": float(box.conf[0]),
                    })

        return person_count, boxes

    def get_crowd_density(self, frame: np.ndarray) -> dict:
        """
        Analyze a frame and return crowd density information.

        Args:
            frame: BGR image as numpy array.

        Returns:
            Dictionary with person_count, density label, and bounding boxes.
        """
        person_count, boxes = self.detect_people(frame)
        density = classify_density(person_count)

        return {
            "person_count": person_count,
            "density": density,
            "density_numeric": density_to_numeric(density),
            "boxes": boxes,
        }

    def process_video(self, video_path: str, sample_rate: int = 30) -> list[dict]:
        """
        Process a video file and return crowd density for sampled frames.

        Args:
            video_path: Path to the video file.
            sample_rate: Process every Nth frame.

        Returns:
            List of crowd density results per sampled frame.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        results = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                density_info = self.get_crowd_density(frame)
                density_info["frame"] = frame_idx
                results.append(density_info)

            frame_idx += 1

        cap.release()
        return results

    def process_webcam(self, duration_seconds: int = 10, sample_rate: int = 30) -> list[dict]:
        """
        Process webcam stream for a given duration.

        Args:
            duration_seconds: How long to capture.
            sample_rate: Process every Nth frame.

        Returns:
            List of crowd density results.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Cannot open webcam")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        max_frames = int(duration_seconds * fps)
        results = []
        frame_idx = 0

        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                density_info = self.get_crowd_density(frame)
                density_info["frame"] = frame_idx
                results.append(density_info)

            frame_idx += 1

        cap.release()
        return results
