import streamlit as st
import pickle
import numpy as np

# Load models
with open("D://logistic_model.pkl", 'rb') as f:
    logistic_data = pickle.load()
logistic_model = logistic_data['model']
logistic_scaler = logistic_data['scaler']

with open("D://svm_model.pkl", 'rb') as f:
    svm_data = pickle.load()
svm_model = svm_data['model']
svm_scaler = svm_data['scaler']

# Streamlit app
st.title("Diabetes Prediction App")

# Inputs
st.header("Enter Patient Details:")
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=80)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=85)
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input("Age", min_value=0, max_value=120, value=30)

# Collect features
features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])

# Model choice
st.header("Choose Prediction Model:")
model_choice = st.radio("Select a model:", ("Logistic Regression", "SVM"))

# Predict
if st.button("Predict"):
    if model_choice == "Logistic Regression":
        scaled_features = logistic_scaler.transform(features)
        prediction = logistic_model.predict(scaled_features)
    else:
        scaled_features = svm_scaler.transform(features)
        prediction = svm_model.predict(scaled_features)

    result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
    st.subheader(f"Prediction: {result}")
