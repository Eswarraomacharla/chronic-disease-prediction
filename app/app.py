import streamlit as st
import joblib
import os
from utils import predict_diabetes, predict_heart_disease

# Set up absolute path to models directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Load models
diabetes_model = joblib.load(os.path.join(MODELS_DIR, 'diabetes_model.pkl'))
heart_model = joblib.load(os.path.join(MODELS_DIR, 'heart_model.pkl'))

# App UI
st.set_page_config(page_title="Chronic Disease Prediction", layout="centered")
st.title("🔬 Chronic Disease Prediction App")

# Sidebar for disease selection
option = st.sidebar.selectbox(
    "Select Disease to Predict:",
    ("Diabetes", "Heart Disease")
)

# Diabetes Prediction
if option == "Diabetes":
    st.header("🩸 Diabetes Prediction")
    pregnancies = st.number_input("Pregnancies", min_value=0)
    glucose = st.number_input("Glucose", min_value=0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0)
    insulin = st.number_input("Insulin", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0)
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    age = st.number_input("Age", min_value=0)

    if st.button("Predict Diabetes"):
        result = predict_diabetes(diabetes_model, [
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, diabetes_pedigree, age
        ])
        st.success(f"Prediction: {'Positive' if result == 1 else 'Negative'}")

# Heart Disease Prediction
elif option == "Heart Disease":
    st.header("❤️ Heart Disease Prediction")
    age = st.number_input("Age", min_value=0)
    sex = st.selectbox("Sex", [0, 1])  # 0 = Female, 1 = Male
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", min_value=0)
    chol = st.number_input("Cholesterol", min_value=0)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Rest ECG", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", min_value=0)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("Oldpeak", min_value=0.0)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thal", [0, 1, 2, 3])

    if st.button("Predict Heart Disease"):
        result = predict_heart_disease(heart_model, [
            age, sex, cp, trestbps, chol, fbs, restecg, thalach,
            exang, oldpeak, slope, ca, thal
        ])
        st.success(f"Prediction: {'Positive' if result == 1 else 'Negative'}")
