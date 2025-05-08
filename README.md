Iris Flower Classification API
This repository contains a FastAPI application that serves a machine learning model trained on the classic Iris dataset. The API allows users to submit sepal and petal measurements and get a prediction of the iris species (Setosa, Versicolor, or Virginica).

Features
FastAPI-based REST API

Scikit-learn RandomForestClassifier model

Model file: iris_model.pkl

Swagger UI for easy testing at /docs

Input validation with Pydantic schemas

Project Structure
ML_deployment/
├── app/
│ ├── init.py
│ ├── main.py # FastAPI app
│ ├── model.py # Model loading and prediction logic
│ └── schemas.py # Pydantic schema for request validation
├── iris_model.pkl # Saved trained ML model
├── requirements.txt # Python dependencies
└── README.md # This file

How to Run
Clone the repository

git clone https://github.com/Rawatkb/MI_deployment.git
cd MI_deployment

Create a virtual environment and install dependencies

python -m venv venv
venv\Scripts\activate (On Windows)
or
source venv/bin/activate (On Mac/Linux)

pip install -r requirements.txt

Run the FastAPI server

uvicorn app.main:app --reload

Test the API

Open your browser and go to http://127.0.0.1:8000/docs
Use the interactive Swagger UI to test the /predict_species endpoint

Example Request
POST /predict_species

{
"sepal_length": 5.1,
"sepal_width": 3.5,
"petal_length": 1.4,
"petal_width": 0.2
}

Response

{
"prediction_index": 0,
"predicted_species": "setosa"
}

Notes
The Iris dataset is used as a demo for machine learning model deployment.
The same pattern can be applied to real-world ML applications such as fraud detection, health diagnosis, or credit scoring.

Author
Rawatkb