import streamlit as st
import pandas as pd, numpy as np, joblib

st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️")
bundle = joblib.load("model.joblib")
pipe, threshold = bundle["pipe"], bundle["threshold"]

st.title("❤️ Heart Disease Risk Predictor")
st.caption("Educational ML project. Not a medical device. Do not use for diagnosis.")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 100, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest pain type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
    trestbps = st.number_input("Resting blood pressure (mm Hg)", 80, 220, 120)
    chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", ["False", "True"])
with col2:
    restecg = st.selectbox("Resting ECG", ["normal", "st-t abnormality", "lv hypertrophy"])
    thalch = st.number_input("Max heart rate achieved", 60, 220, 150)
    exang = st.selectbox("Exercise-induced angina", ["False", "True"])
    oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1)
    slope = st.selectbox("ST slope", ["upsloping", "flat", "downsloping"])
    ca = st.selectbox("Major vessels colored (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversable defect"])

if st.button("Predict"):
    row = pd.DataFrame([{"age": age, "trestbps": trestbps, "chol": chol, "thalch": thalch,
                         "oldpeak": oldpeak, "ca": ca, "sex": sex, "cp": cp, "fbs": fbs,
                         "restecg": restecg, "exang": exang, "slope": slope, "thal": thal}])
    prob = pipe.predict_proba(row)[0, 1]
    st.metric("Estimated risk", f"{prob:.0%}")
    if prob >= threshold:
        st.error("⚠️ Higher risk. A doctor should review this.")
    else:
        st.success("✅ Lower risk based on the entered values.")