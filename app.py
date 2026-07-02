import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Student Performance", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS FOR UI STYLING ---
st.markdown("""
<style>
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

# --- CACHE DATA LOADING ---
@st.cache_data
def load_data():
    try:
        # Tries to load the dataset for the analytics page
        return pd.read_csv('student_performance.csv')
    except FileNotFoundError:
        try:
            return pd.read_csv('Dataset/student_performance.csv')
        except FileNotFoundError:
            return None

df = load_data()

# --- LOAD MODEL ---
try:
    with open('best_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found! Please ensure 'best_model.pkl' is in the repository.")
    model = None

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Home", "🎯 Single Prediction", "📊 Data Analytics", "ℹ️ About Project"])

# --- PAGE 1: HOME ---
if page == "🏠 Home":
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
            
            # POST-RESULT ANALYSIS CHART
            st.subheader("📊 Performance Analysis")
            if df is not None:
                avg_score = df['Performance_Index'].mean()
                
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(['Average Student', 'This Student (Predicted)'], [avg_score, prediction], color=['#1A365D', '#00A896'])
                ax.set_ylabel('Performance Index')
                ax.set_ylim(0, 100)
                for i, v in enumerate([avg_score, prediction]):
                    ax.text(i, v + 2, f"{v:.1f}", ha='center', fontweight='bold')
                st.pyplot(fig)
            else:
                st.info("Upload dataset to view comparison graphs.")
        else:
            st.error("Model is not loaded. Cannot make predictions.")

# --- PAGE 3: DATA ANALYTICS (NEW!) ---
elif page == "📊 Data Analytics":
    st.title("📊 Exploratory Data Analysis")
    st.write("Visualizing the relationships between study habits and student performance.")
    
    if df is not None:
        # Create tabs for organized viewing
        tab1, tab2, tab3, tab4 = st.tabs(["📉 Histogram", "📐 Scatter Plot", "📦 Boxplot", "🌡️ Heatmap"])
        
        # Plot styling for dark mode
        plt.style.use('dark_background')
        
        with tab1:
            st.subheader("Distribution of Performance Scores")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(df['Performance_Index'], bins=30, kde=True, color='#00A896', ax=ax)
            ax.set_xlabel("Performance Index")
            st.pyplot(fig)
            
        with tab2:
            st.subheader("Hours Studied vs. Performance")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.scatterplot(data=df, x='Hours_Studied', y='Performance_Index', alpha=0.6, color='#1A365D', ax=ax)
            ax.set_xlabel("Hours Studied")
            ax.set_ylabel("Performance Index")
            st.pyplot(fig)
            
        with tab3:
            st.subheader("Impact of Extracurricular Activities")
            fig, ax = plt.subplots(figsize=(10, 5))
            # Handle encoded vs unencoded data gracefully
            if df['Extracurricular_Activities'].dtype in ['int64', 'int32']:
                df['Extra_Label'] = df['Extracurricular_Activities'].map({1: 'Yes', 0: 'No'})
                sns.boxplot(data=df, x='Extra_Label', y='Performance_Index', palette="Set2", ax=ax)
                ax.set_xlabel("Participates in Extracurriculars")
            else:
                sns.boxplot(data=df, x='Extracurricular_Activities', y='Performance_Index', palette="Set2", ax=ax)
            st.pyplot(fig)
            
        with tab4:
            st.subheader("Feature Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            # Only correlate numerical columns
            numeric_df = df.select_dtypes(include=[np.number])
            sns.heatmap(numeric_df.corr(), annot=True, cmap="mako", fmt=".2f", ax=ax)
            st.pyplot(fig)
            
    else:
        st.warning("⚠️ Dataset not found. Please upload 'student_performance.csv' to the GitHub repository to view analytics.")

# --- PAGE 4: ABOUT PROJECT ---
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
