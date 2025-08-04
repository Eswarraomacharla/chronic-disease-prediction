import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

DATA_PATH = '../data/heart.csv'
MODEL_PATH = '../models/heart_model.pkl'

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y

def main():
    df = load_data(DATA_PATH)
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    print("Model saved at:", MODEL_PATH)

if __name__ == "__main__":
    main()
