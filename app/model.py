"""
Model loading and prediction module.
"""

import joblib
import numpy as np

# Load the trained iris classification model
try:
    model = joblib.load("iris_model.pkl")
except FileNotFoundError as e:
    raise RuntimeError(
        "Model file not found. Please ensure 'iris_model.pkl' exists."
    ) from e
except Exception as e:
    raise RuntimeError(f"An error occurred while loading the model: {e}") from e


def predict(features: list) -> np.ndarray:
    """
    Make a prediction based on the input features.

    Args:
        features (list): A list of input features in the order:
                         [sepal_length, sepal_width, petal_length, petal_width]

    Returns:
        np.ndarray: The predicted class index.
    """
    input_data = np.array([features])
    prediction = model.predict(input_data)
    return prediction
