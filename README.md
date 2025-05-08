# Iris Flower Classification API

This repository contains a FastAPI application that serves a machine learning model trained on the classic Iris dataset.
The API allows users to submit sepal and petal measurements and get a prediction of the iris species (Setosa, Versicolor, or Virginica).

## Features

* FastAPI-based REST API
* Scikit-learn RandomForestClassifier model
* Model file: iris\_model.pkl
* Swagger UI for easy testing at /docs
* Input validation with Pydantic schemas

## Project Structure

ML\_deployment/
├── app/
│   ├── **init**.py
│   ├── main.py        # FastAPI app
│   ├── model.py       # Model loading and prediction logic
│   └── schemas.py     # Pydantic schema for request validation
├── iris\_model.pkl     # Saved trained ML model
├── requirements.txt   # Python dependencies
└── README.md          # This file

## How to Run

### 1. Clone the repository

git clone [https://github.com/Rawatkb/MI\_deployment.git](https://github.com/Rawatkb/MI_deployment.git)
cd MI\_deployment

### 2. Create a virtual environment and install dependencies

python -m venv venv
venv\Scripts\activate    (On Windows)
or
source venv/bin/activate (On Mac/Linux)

pip install -r requirements.txt

### 3. Run the FastAPI server

uvicorn app.main\:app --reload

### 4. Test the API

Open your browser and go to:
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
Use the Swagger UI to test the /predict\_species endpoint.

## Example Request

POST /predict\_species

{
"sepal\_length": 5.1,
"sepal\_width": 3.5,
"petal\_length": 1.4,
"petal\_width": 0.2
}

Response:

{
"prediction\_index": 0,
"predicted\_species": "setosa"
}

## Notes

This project is for educational purposes to demonstrate machine learning model deployment with FastAPI.
The same deployment structure can be adapted to real-world applications such as fraud detection, medical diagnosis, and more.

## Author

Rawatkb


