import subprocess
import sys

# Install requirements.txt packages if not already installed
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

import streamlit as st
import numpy as np
from joblib import load


# Load the trained model
model = load('linear_regression_model.joblib')

st.title("🌍 Earthquake Significance Prediction")

# User input
user_input = st.number_input("Enter Significance Value", min_value=0.0, step=1.0)

if st.button("Predict"):
    if user_input < 650:
        st.success(f"Significance: {user_input}\n✅ Status: Normal")
    elif user_input > 2910:
        st.warning(f"Significance: {user_input}\n❓ Status: Unpredictable")
    else:
        # Prepare input for prediction
        input_array = np.array([user_input]).reshape(-1, 1)
        predicted_magnitude = model.predict(input_array)
        pred_value = float(predicted_magnitude.flatten()[0])

        st.info(f"Significance: {user_input}")
        st.write(f"Predicted Magnitude: **{pred_value:.2f}**")
        st.error("Status: Significant")
