
import pandas as pd
import numpy as np

def preprocess_diabetes_input(data: dict) -> pd.DataFrame:
    """
    Preprocess input data for diabetes prediction.
    Assumes `data` is a dictionary with keys matching the feature names expected by the model.
    """
    expected_features = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ]

    # Ensure order and type correctness
    input_data = pd.DataFrame([data], columns=expected_features)
    return input_data

def preprocess_heart_input(data: dict) -> pd.DataFrame:
    """
    Preprocess input data for heart disease prediction.
    """
    expected_features = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]

    # Ensure order and type correctness
    input_data = pd.DataFrame([data], columns=expected_features)
    return input_data
