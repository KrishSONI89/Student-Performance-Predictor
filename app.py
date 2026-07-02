import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Student Performance", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS FOR UI STYLING ---
# This block injects custom HTML/CSS to create the beautiful gradient hero section and metric cards
st.markdown("""
<style>
    /* Hero Banner Styling */
    .hero-container {
        background: linear-gradient(135deg, #0B192C 0%, #1A365D 50%, #00A896 100%);
        border-radius: 24px;
        padding: 60px 40px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        margin-bottom: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 10px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .hero-subtitle {
        font-size: 1.4rem;
        font-weight: 300;
        opacity: 0.9;
        margin-bottom: 25px;
    }
    .hero-badge-container {
        display: flex;
        justify-content: center;
        gap: 15px;
    }
    .hero-badge {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        padding: 8px 24px;
        border-radius: 50px;
        font-size: 0.85rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Metric Cards Styling */
    .metrics-wrapper {
        display: flex;
        justify-content: space-between;
        gap: 20px;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #1E2130;
        border-radius: 16px;
        padding: 24px;
        flex: 1;
        border: 1px solid #2D324A;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        color: white;
    }
    .metric-header {
        font-size: 0.9rem;
        color: #A0A5C0;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD MODEL ---
try:
    with open('best_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found! Please ensure 'best_model.pkl' is in the repository.")
    model = None

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Home", "🎯 Single Prediction", "ℹ️ About Project"])

# --- PAGE 1: HOME ---
if page == "🏠 Home":
    
    # 1. Custom Hero Banner
    st.markdown("""
        <div class="hero-container">
            <div style="font-size: 3rem; margin-bottom: 10px;">🎓</div>
            <div class="hero-title">Student Performance AI</div>
            <div class="hero-subtitle">Smart Academic Intelligence & Prediction Engine</div>
            <div class="hero-badge-container">
                <div class="hero-badge">PREDICT</div>
                <div class="hero-badge">ANALYZE</div>
                <div class="hero-badge">IMPROVE</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # 2. Custom Metric Cards (Matching the 4-box layout)
    st.markdown("""
        <div class="metrics-wrapper">
            <div class="metric-card">
                <div class="metric-header">👨‍🎓 Data Baseline</div>
                <div class="metric-value">5,000</div>
            </div>
            <div class="metric-card">
                <div class="metric-header">🎯 Model Accuracy (R2)</div>
                <div class="metric-value">91%</div>
            </div>
            <div class="metric-card">
                <div class="metric-header">📊 Features Analyzed</div>
                <div class="metric-value">5</div>
            </div>
            <div class="metric-card">
                <div class="metric-header">⚡ Pipeline Status</div>
                <div class="metric-value">Real-Time</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# --- PAGE 2: SINGLE PREDICTION ---
elif page == "🎯 Single Prediction":
    st.title("🎯 Predict Student Performance")
    st.write("Enter the student's details below to predict their academic performance index.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        hours_studied = st.number_input("Hours Studied per Day", min_value=1, max_value=24, value=5)
        previous_scores = st.number_input("Previous Scores (Out of 100)", min_value=0, max_value=100, value=70)
        extracurricular = st.selectbox("Participates in Extracurricular Activities?", ["Yes", "No"])
        
    with col2:
        sleep_hours = st.number_input("Hours of Sleep per Day", min_value=1, max_value=24, value=7)
        sample_papers = st.number_input("Sample Question Papers Practiced", min_value=0, max_value=100, value=3)

    extra_encoded = 1 if extracurricular == "Yes" else 0
    st.markdown("---")
    
    if st.button("Predict Performance Index", use_container_width=True):
        if model:
            input_data = pd.DataFrame({
                'Hours_Studied': [hours_studied],
                'Previous_Scores': [previous_scores],
                'Extracurricular_Activities': [extra_encoded],
                'Sleep_Hours': [sleep_hours],
                'Sample_Question_Papers_Practiced': [sample_papers]
            })
            
            prediction = model.predict(input_data)[0]
            prediction = np.clip(prediction, 10, 100) 
            
            st.success(f"📈 **Predicted Performance Index: {prediction:.2f} / 100**")
        else:
            st.error("Model is not loaded. Cannot make predictions.")

# --- PAGE 3: ABOUT PROJECT ---
elif page == "ℹ️ About Project":
    st.title("ℹ️ About This Project")
    st.markdown("""
    ### 💻 Machine Learning Pipeline
    This project was built following a strict Machine Learning lifecycle:
    
    1. **Data Preprocessing:** Handled missing values and encoded categorical data.
    2. **Exploratory Data Analysis (EDA):** Analyzed correlations between study habits and scores.
    3. **Model Building:** Trained multiple regression models including Linear Regression, Random Forest, and XGBoost.
    4. **Deployment:** Hosted live on Streamlit Community Cloud.
    
    ***
    *Created as part of the AIML Summer Internship Capstone Project.*
    """)
