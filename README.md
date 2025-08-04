# Chronic Disease Prediction Web App

This project is a web-based application that predicts the likelihood of two major chronic diseases — **Diabetes** and **Heart Disease** — using machine learning models. It is built using **Streamlit** for the frontend interface and **scikit-learn** for machine learning. The goal is to provide early predictions based on input symptoms and medical attributes to assist in preliminary medical screening.

---

## 🔍 Features

- **Diabetes Prediction**: Predicts the likelihood of diabetes based on input such as glucose level, BMI, pregnancies, etc.
- **Heart Disease Prediction**: Predicts the possibility of heart disease based on factors like age, blood pressure, cholesterol, etc.
- Clean and interactive **Streamlit UI**
- Pre-trained models loaded using **joblib**
- Simple and intuitive input forms

---

## 🗂️ Project Structure

chronic_disease_prediction/
│
├── app/
│ ├── app.py # Main Streamlit application
│ ├── utils.py # Utility functions for preprocessing
│ └── init.py # Package initializer
│
├── models/
│ ├── diabetes_model.pkl # Pre-trained diabetes model
│ └── heart_model.pkl # Pre-trained heart disease model
│
├── requirements.txt # Required Python packages
└── README.md # Project overview and instructions


---

## ⚙️ How It Works

### 1. Input Collection
Users are asked to enter their medical details through a user-friendly web interface powered by Streamlit.

### 2. Data Preprocessing
Input is cleaned, scaled (if needed), and prepared using helper functions from `utils.py`.

### 3. Prediction
Pre-trained machine learning models are used to make predictions, which are displayed to the user instantly.

---

## 🚀 How to Run

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/chronic_disease_prediction.git
cd chronic_disease_prediction
### Step 2: Create a Virtual Environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

