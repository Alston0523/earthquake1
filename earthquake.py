import streamlit as st
import folium
from streamlit_folium import st_folium
from joblib import load
import numpy as np
import reverse_geocode as rg

# Load trained model
model = load('linear_regression_model.joblib')

# Function to get country from lat/lon
def get_country(lat, lon):
    result = rg.search([(lat, lon)])
    if result and 'country' in result[0]:
        return result[0]['country']  # return 2-letter code
    return "Unknown"

st.title("🌍 Earthquake Prediction System")

# Step 1: Create interactive map
st.subheader("Click on the map to select location")

m = folium.Map(location=[0, 0], zoom_start=2)
m.add_child(folium.LatLngPopup())

map_data = st_folium(m, width=700, height=500)

# Step 2: Show clicked coordinates
if map_data and map_data["last_clicked"]:
    latitude = map_data["last_clicked"]["lat"]
    longitude = map_data["last_clicked"]["lng"]

    st.success(f"📍 Selected Location: ({latitude:.4f}, {longitude:.4f})")

    # Step 3: Get earthquake parameters from user with validation
    sig = st.number_input("Enter significance:", min_value=0.0, step=1.0)
    depth = st.number_input("Enter depth (km):", min_value=0.0, step=0.1)
    gap = st.number_input("Enter gap (degrees):", min_value=0.0, max_value=360.0, step=0.1)

    # Step 4: Predict button
    if st.button("Predict Earthquake Magnitude"):
        # --- Input validation ---
        invalid_inputs = []
        if sig < 0:
            invalid_inputs.append("Significance must be >= 0")
        if depth < 0:
            invalid_inputs.append("Depth must be >= 0 km")
        if gap < 0 or gap > 360:
            invalid_inputs.append("Gap must be between 0 and 360°")
        if latitude < -90 or latitude > 90:
            invalid_inputs.append("Latitude must be between -90 and 90°")
        if longitude < -180 or longitude > 180:
            invalid_inputs.append("Longitude must be between -180 and 180°")

        if invalid_inputs:
            st.error("⚠️ Please fix the following inputs before predicting:")
            for msg in invalid_inputs:
                st.write(f"- {msg}")
        else:
            # --- Significance check ---
            if sig < 650:
                sig_status = "Normal"
            elif sig > 2910:
                sig_status = "Unpredictable"
            else:
                sig_status = "Within prediction range"

            # --- Depth check ---
            if depth < 2.6:
                depth_status = "Normal (low impact)"
            elif depth > 671:
                depth_status = "Unpredictable depth"
            elif depth <= 70:
                depth_status = "Shallow (more dangerous)"
            else:
                depth_status = "Deep (less surface impact)"

            # --- Gap check ---
            if gap <= 90:
                gap_status = "Highly reliable location (very significant)"
            elif gap <= 180:
                gap_status = "Reliable location (significant)"
            elif gap <= 270:
                gap_status = "Low reliability (less significant)"
            else:
                gap_status = "Unreliable location estimate (low significance)"

            # --- Prediction ---
            if sig_status == "Within prediction range" and "Unpredictable" not in depth_status:
                try:
                    input_array = np.array([[sig, depth, gap, latitude, longitude]])
                    predicted_magnitude = model.predict(input_array)
                    pred_value = float(predicted_magnitude.flatten()[0])

                    st.subheader("🔍 Earthquake Prediction Result")
                    st.write(f"**Significance:** {sig} → {sig_status}")
                    st.write(f"**Depth:** {depth} km → {depth_status}")
                    st.write(f"**Gap:** {gap}° → {gap_status}")
                    st.write(f"**Location:** ({latitude:.4f}, {longitude:.4f}), {get_country(latitude, longitude)}")
                    st.success(f"**Predicted Magnitude:** {pred_value:.2f}")
                except ValueError as e:
                    st.error(f"Model input error: {e}")
            else:
                st.warning("Prediction skipped due to out-of-range values.")
else:
    st.info("Click anywhere on the map to select a location.")
