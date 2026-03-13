"""
Module 3 — Risk Prediction Model.

Uses scikit-learn RandomForestClassifier to predict safety risk scores.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODEL_PATH = os.path.join(DATA_DIR, "risk_model.pkl")
DATASET_PATH = os.path.join(DATA_DIR, "safety_dataset.csv")


def generate_sample_dataset(n_samples: int = 2000, save: bool = True) -> pd.DataFrame:
    """
    Generate a synthetic safety dataset for training.

    Features:
        - time_of_day (0-23)
        - crowd_density (0=low, 1=medium, 2=high)
        - lighting_level (0=dark, 1=dim, 2=well-lit)
        - crime_history (0-5)

    Target:
        - risk_score (0-10)

    The risk score is computed using a realistic heuristic and then
    discretized into integer classes.
    """
    rng = np.random.RandomState(42)

    time_of_day = rng.randint(0, 24, n_samples)
    crowd_density = rng.choice([0, 1, 2], n_samples, p=[0.4, 0.35, 0.25])
    lighting_level = rng.choice([0, 1, 2], n_samples, p=[0.3, 0.3, 0.4])
    crime_history = rng.randint(0, 6, n_samples)

    # Compute risk score using a heuristic formula
    # Higher risk when: late at night, low crowd, poor lighting, high crime
    time_risk = np.where(
        (time_of_day >= 22) | (time_of_day <= 5),
        3.0,
        np.where((time_of_day >= 18) | (time_of_day <= 7), 1.5, 0.5),
    )

    crowd_risk = np.where(crowd_density == 0, 2.0, np.where(crowd_density == 1, 0.5, 1.0))
    lighting_risk = np.where(lighting_level == 0, 2.5, np.where(lighting_level == 1, 1.0, 0.0))
    crime_risk = crime_history * 0.6

    raw_risk = time_risk + crowd_risk + lighting_risk + crime_risk
    noise = rng.normal(0, 0.3, n_samples)
    raw_risk = raw_risk + noise

    # Normalize to 0-10 range
    risk_min, risk_max = raw_risk.min(), raw_risk.max()
    risk_score = ((raw_risk - risk_min) / (risk_max - risk_min)) * 10
    risk_score = np.clip(np.round(risk_score), 0, 10).astype(int)

    df = pd.DataFrame({
        "time_of_day": time_of_day,
        "crowd_density": crowd_density,
        "lighting_level": lighting_level,
        "crime_history": crime_history,
        "risk_score": risk_score,
    })

    if save:
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(DATASET_PATH, index=False)
        print(f"Dataset saved to {DATASET_PATH} ({n_samples} samples)")

    return df


def load_dataset(path: str | None = None) -> pd.DataFrame:
    """Load the safety dataset from CSV."""
    path = path or DATASET_PATH
    if not os.path.exists(path):
        print("Dataset not found. Generating sample dataset...")
        return generate_sample_dataset()
    return pd.read_csv(path)


def train_model(df: pd.DataFrame | None = None, save: bool = True) -> RandomForestClassifier:
    """
    Train the RandomForestClassifier risk prediction model.

    Args:
        df: Training dataframe. If None, loads from disk.
        save: Whether to save the trained model.

    Returns:
        Trained RandomForestClassifier.
    """
    if df is None:
        df = load_dataset()

    feature_cols = ["time_of_day", "crowd_density", "lighting_level", "crime_history"]
    X = df[feature_cols]
    y = df["risk_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print(f"Model accuracy: {accuracy:.2%}")
    print(f"Feature importances: {dict(zip(feature_cols, model.feature_importances_.round(3)))}")

    if save:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        print(f"Model saved to {MODEL_PATH}")

    return model


def load_model(path: str | None = None) -> RandomForestClassifier:
    """Load a trained model from disk."""
    path = path or MODEL_PATH
    if not os.path.exists(path):
        print("Trained model not found. Training new model...")
        return train_model()

    with open(path, "rb") as f:
        return pickle.load(f)


def predict_risk(
    model: RandomForestClassifier,
    time_of_day: int,
    crowd_density: int,
    lighting_level: int,
    crime_history: int,
) -> float:
    """
    Predict risk score for given features.

    Args:
        model: Trained RandomForestClassifier.
        time_of_day: Hour (0-23).
        crowd_density: 0=low, 1=medium, 2=high.
        lighting_level: 0=dark, 1=dim, 2=well-lit.
        crime_history: Crime score (0-5).

    Returns:
        Risk score as a float (0.0-10.0), using probability-weighted average
        for finer granularity than integer classes.
    """
    feature_names = ["time_of_day", "crowd_density", "lighting_level", "crime_history"]
    features = pd.DataFrame(
        [[time_of_day, crowd_density, lighting_level, crime_history]],
        columns=feature_names,
    )
    probabilities = model.predict_proba(features)[0]
    classes = model.classes_

    # Weighted average of class probabilities for a continuous score
    risk_score = float(np.dot(probabilities, classes))
    return round(risk_score, 1)


if __name__ == "__main__":
    print("=== SafeRoute AI - Risk Model Training ===\n")
    df = generate_sample_dataset()
    model = train_model(df)

    print("\n--- Sample Predictions ---")
    test_cases = [
        {"time_of_day": 14, "crowd_density": 2, "lighting_level": 2, "crime_history": 0},
        {"time_of_day": 23, "crowd_density": 0, "lighting_level": 0, "crime_history": 4},
        {"time_of_day": 8, "crowd_density": 1, "lighting_level": 2, "crime_history": 1},
    ]

    for case in test_cases:
        score = predict_risk(model, **case)
        print(f"  Features: {case} → Risk Score: {score}")
