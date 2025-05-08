"""
Main FastAPI app to serve the Iris flower classification.
"""

from fastapi import FastAPI
from app import model, schemas
from sklearn.datasets import load_iris

# Load iris target names
iris_data = load_iris()
target_names = iris_data.target_names

app = FastAPI(
    title="Iris Flower Classification API",
    description="Predict the species of an iris flower using a trained ML model.",
    version="1.0.0",
)


@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Flower Prediction API 🌸"}


@app.post("/predict_species")
def predict_species(sample: schemas.IrisSample):
    """
    Predict the iris species based on sepal and petal measurements.
    """
    features = [
        sample.sepal_length,
        sample.sepal_width,
        sample.petal_length,
        sample.petal_width,
    ]
    prediction = model.predict(features)
    predicted_species = target_names[prediction[0]]
    return {
        "prediction_index": int(prediction[0]),
        "predicted_species": predicted_species
    }
