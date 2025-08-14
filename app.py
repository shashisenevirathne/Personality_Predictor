import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models and encoders (files should be in the same folder as app.py)
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')
target_encoder = joblib.load('target_encoder.pkl')

# Streamlit Page Config
st.set_page_config(page_title="Personality Predictor", page_icon="🧠", layout="centered")

# Custom CSS Styling
st.markdown("""
    <style>
    .main-title {
        font-size: 46px;
        font-weight: 800;
        color: #1A5276;
        text-align: center;
        padding-top: 20px;
    }
    .subtitle {
        font-size: 20px;
        color: #138D75;
        text-align: center;
        margin-bottom: 30px;
    }
    .result {
        background-color: #D1F2EB;
        padding: 20px;
        border-radius: 10px;
        font-size: 24px;
        color: #117864;
        text-align: center;
        font-weight: bold;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #2980B9;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1F618D;
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)

# Title Section
st.markdown('<div class="main-title">🧠 Personality Type Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Classify whether someone is an Introvert or Extrovert based on behavior patterns</div>', unsafe_allow_html=True)

# Input Section
st.markdown("#### Please enter your behavioral traits:")

col1, col2 = st.columns(2)

with col1:
    time_alone = st.slider("🕒 Time Spent Alone (Hours/Day)", 0, 10, 3)
    social_events = st.slider("🎉 Social Events Attended per Week", 0, 10, 3)
    friends_size = st.slider("👥 Friends Circle Size", 0, 20, 10)
    post_freq = st.slider("📱 Social Media Posts per Week", 0, 10, 5)

with col2:
    stage_fear = st.selectbox("🎤 Stage Fear?", ['Yes', 'No'])
    going_outside = st.slider("🌳 Days Going Outside per Week", 0, 7, 3)
    drained_social = st.selectbox("😮‍ Drained After Socializing?", ['Yes', 'No'])

# Prediction Section
try:
    input_data = {
        'Time_spent_Alone': time_alone,
        'Stage_fear': label_encoders['Stage_fear'].transform([stage_fear])[0],
        'Social_event_attendance': social_events,
        'Going_outside': going_outside,
        'Drained_after_socializing': label_encoders['Drained_after_socializing'].transform([drained_social])[0],
        'Friends_circle_size': friends_size,
        'Post_frequency': post_freq
    }

    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)

    if st.button("🔍 Predict Personality"):
        prediction = model.predict(input_scaled)
        personality = target_encoder.inverse_transform(prediction)[0]

        st.markdown(f"""<div class="result"> You are likely an **{personality.upper()}**</div>""", unsafe_allow_html=True)

except Exception as e:
    st.error(f"🚫 An error occurred during prediction:\n\n{e}")
