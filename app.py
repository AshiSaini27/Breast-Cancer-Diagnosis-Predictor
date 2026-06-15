import streamlit as st
import joblib
from sklearn.datasets import load_breast_cancer

# Load model
model = joblib.load("model.pkl")

data = load_breast_cancer()

st.title("Breast Cancer Diagnosis Predictor")

st.write("Enter feature values below:")

inputs = []

for feature in data.feature_names:
    value = st.number_input(feature, value=0.0)
    inputs.append(value)


if st.button("Predict"):

    prediction = model.predict([inputs])[0]
    probability = model.predict_proba([inputs])[0]

    confidence = max(probability) * 100

    if prediction == 0:
        st.error("Malignant")
        st.write(f"Confidence: {confidence:.2f}%")
    else:
        st.success("Benign")
        st.write(f"Confidence: {confidence:.2f}%")