"""
Generate a sample test video with simulated people (rectangles) for
crowd detection testing.
"""

import os
import cv2
import numpy as np


def generate_sample_video(
    output_path: str = None,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    duration_seconds: int = 5,
) -> str:
    """
    Generate a sample video with moving rectangles simulating people.

    Args:
        output_path: Output video file path.
        width: Frame width.
        height: Frame height.
        fps: Frames per second.
        duration_seconds: Video duration in seconds.

    Returns:
        Path to the generated video file.
    """
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "sample_video.avi")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = fps * duration_seconds
    rng = np.random.RandomState(42)

    # Create simulated "people" as colored rectangles
    num_people = 8
    people = []
    for _ in range(num_people):
        x = rng.randint(50, width - 80)
        y = rng.randint(50, height - 150)
        w = rng.randint(30, 60)
        h = rng.randint(80, 140)
        color = tuple(int(c) for c in rng.randint(100, 255, 3))
        dx = rng.choice([-2, -1, 1, 2])
        dy = rng.choice([-1, 0, 1])
        people.append({"x": x, "y": y, "w": w, "h": h, "color": color, "dx": dx, "dy": dy})

    for frame_idx in range(total_frames):
        # Create a background (urban street scene simulation)
        frame = np.full((height, width, 3), (180, 180, 180), dtype=np.uint8)

        # Draw road
        cv2.rectangle(frame, (0, height // 3), (width, 2 * height // 3), (100, 100, 100), -1)
        # Draw road lines
        for x_pos in range(0, width, 40):
            cv2.rectangle(
                frame,
                (x_pos, height // 2 - 2),
                (x_pos + 20, height // 2 + 2),
                (255, 255, 255),
                -1,
            )

        # Draw and move people
        for person in people:
            # Draw body (rectangle)
            x, y, w, h = person["x"], person["y"], person["w"], person["h"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), person["color"], -1)
            # Draw head (circle)
            head_radius = w // 3
            cv2.circle(frame, (x + w // 2, y - head_radius), head_radius, person["color"], -1)

            # Move person
            person["x"] += person["dx"]
            person["y"] += person["dy"]

            # Bounce off edges
            if person["x"] <= 0 or person["x"] + person["w"] >= width:
                person["dx"] *= -1
            if person["y"] <= 0 or person["y"] + person["h"] >= height:
                person["dy"] *= -1

        # Add frame number text
        cv2.putText(
            frame,
            f"Frame: {frame_idx}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )

        out.write(frame)

    out.release()
    print(f"Sample video generated: {output_path}")
    print(f"  Resolution: {width}x{height}, FPS: {fps}, Duration: {duration_seconds}s")
    return output_path


if __name__ == "__main__":
    generate_sample_video()
