import streamlit as st
import joblib
from utils import preprocess_diabetes_input, preprocess_heart_input

# Load models
diabetes_model = joblib.load("../models/diabetes_model.pkl")
heart_model = joblib.load("D:/chronic disease prediction/chronic_disease_prediction/models/heart_model.pkl")

# Sidebar for selection
st.sidebar.title("Disease Prediction App")
option = st.sidebar.selectbox("Select Disease", ["Diabetes", "Heart Disease"])

st.title(f"{option} Prediction")

if option == "Diabetes":
    pregnancies = st.number_input("Pregnancies", 0)
    glucose = st.number_input("Glucose", 0)
    bp = st.number_input("Blood Pressure", 0)
    skin = st.number_input("Skin Thickness", 0)
    insulin = st.number_input("Insulin", 0)
    bmi = st.number_input("BMI", 0.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0)
    age = st.number_input("Age", 0)

    input_dict = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": bp,
        "SkinThickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }

    if st.button("Predict"):
        input_df = preprocess_diabetes_input(input_dict)
        result = diabetes_model.predict(input_df)[0]
        st.success("Positive for Diabetes" if result == 1 else "Negative for Diabetes")

elif option == "Heart Disease":
    age = st.number_input("Age", 0)
    sex = st.selectbox("Sex", [0, 1])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 0)
    chol = st.number_input("Cholesterol", 0)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", 0)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST depression", 0.0)
    slope = st.selectbox("Slope (0-2)", [0, 1, 2])
    ca = st.selectbox("Number of vessels (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thal (1 = normal; 2 = fixed defect; 3 = reversable defect)", [0, 1, 2, 3])

    input_dict = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }

    if st.button("Predict"):
        input_df = preprocess_heart_input(input_dict)
        result = heart_model.predict(input_df)[0]
        st.success("Positive for Heart Disease" if result == 1 else "Negative for Heart Disease")
