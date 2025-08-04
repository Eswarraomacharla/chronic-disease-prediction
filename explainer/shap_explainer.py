import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load diabetes model and data
model = joblib.load("../models/diabetes_model.pkl")
data = pd.read_csv("../data/diabetes.csv").drop("Outcome", axis=1)

# Explain model
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(data)

# Plot summary
shap.summary_plot(shap_values[1], data)
