"""
Module 2 — Feature Extraction.

Converts environmental signals into ML features for risk prediction.
"""

from datetime import datetime


def estimate_lighting_level(hour: int) -> int:
    """
    Estimate lighting level based on time of day.

    Returns:
        0 = dark, 1 = dim, 2 = well-lit
    """
    if 6 <= hour < 8 or 18 <= hour < 20:
        return 1  # dim (dawn/dusk)
    elif 8 <= hour < 18:
        return 2  # well-lit (daytime)
    else:
        return 0  # dark (night)


def get_historical_crime_score(location: str, crime_data: dict | None = None) -> int:
    """
    Get historical crime score for a location.

    Args:
        location: Identifier for the road segment.
        crime_data: Optional dictionary mapping locations to crime scores.

    Returns:
        Crime score from 0 (safe) to 5 (very dangerous).
    """
    default_crime_data = {
        "road_segment_1": 1,
        "road_segment_2": 0,
        "road_segment_3": 3,
        "road_segment_4": 2,
        "road_segment_5": 4,
        "road_segment_6": 1,
        "road_segment_7": 5,
        "road_segment_8": 2,
        "road_segment_9": 3,
        "road_segment_10": 0,
        "road_segment_11": 4,
        "road_segment_12": 2,
    }

    data = crime_data if crime_data is not None else default_crime_data
    return data.get(location, 1)


def extract_features(
    crowd_density: int,
    location: str = "road_segment_1",
    timestamp: datetime | None = None,
    crime_data: dict | None = None,
) -> dict:
    """
    Extract ML features from environmental signals.

    Args:
        crowd_density: Numeric crowd density (0=low, 1=medium, 2=high).
        location: Road segment identifier.
        timestamp: Current time (defaults to now).
        crime_data: Optional crime score mapping.

    Returns:
        Feature dictionary for the ML model.

    Example output:
        {
            "time_of_day": 23,
            "crowd_density": 0,
            "lighting_level": 1,
            "crime_history": 2
        }
    """
    if timestamp is None:
        timestamp = datetime.now()

    hour = timestamp.hour

    return {
        "time_of_day": hour,
        "crowd_density": crowd_density,
        "lighting_level": estimate_lighting_level(hour),
        "crime_history": get_historical_crime_score(location, crime_data),
    }
