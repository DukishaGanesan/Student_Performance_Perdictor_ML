import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🎓 Student Performance Predictor")

st.write("Enter student details to predict final score")

hours = st.slider("Hours Studied", 1, 10, 5)
attendance = st.slider("Attendance (%)", 50, 100, 80)
previous = st.slider("Previous Score", 40, 100, 70)
sleep = st.slider("Sleep Hours", 4, 9, 6)

if st.button("Predict"):
    input_data = np.array([[hours, attendance, previous, sleep]])
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    
    st.success(f"Predicted Final Score: {prediction[0]:.2f}")
