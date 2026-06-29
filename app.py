import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Student Performance Prediction System")
st.write("Predict student academic performance based on study habits and previous scores.")

# User Inputs
hours_studied = st.number_input("Hours Studied per Day", min_value=1, max_value=24, value=5)
previous_scores = st.number_input("Previous Scores (Out of 100)", min_value=0, max_value=100, value=70)
extracurricular = st.selectbox("Participates in Extracurricular Activities?", ["Yes", "No"])
sleep_hours = st.number_input("Hours of Sleep per Day", min_value=1, max_value=24, value=7)
sample_papers = st.number_input("Sample Question Papers Practiced", min_value=0, max_value=100, value=3)

# Encoding logic matching the training phase
extra_encoded = 1 if extracurricular == "Yes" else 0

if st.button("Predict Performance Index"):
    input_data = pd.DataFrame({
        'Hours_Studied': [hours_studied],
        'Previous_Scores': [previous_scores],
        'Extracurricular_Activities': [extra_encoded],
        'Sleep_Hours': [sleep_hours],
        'Sample_Question_Papers_Practiced': [sample_papers]
    })
    
    prediction = model.predict(input_data)[0]
    prediction = np.clip(prediction, 10, 100) # Ensure it stays within logical bounds
    
    st.success(f"Predicted Performance Index: {prediction:.2f}/100")
