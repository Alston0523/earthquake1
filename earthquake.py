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
    result = rg.get((lat, lon))
    return result['country']

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

    # Step 3: Get earthquake parameters as text inputs
    sig_input = st.text_input("Enter significance [1 - 2910]:", placeholder="Enter significance")
    depth_input = st.text_input("Enter depth (km) [1.0 - 671.0]:", placeholder="Enter depth")
    gap_input = st.text_input("Enter gap (degrees) [0 - 365]:", placeholder="Enter gap")

    # Step 4: Predict button
    if st.button("Predict Earthquake Magnitude"):

        # --- Validation ---
        try:
            sig = float(sig_input)
            depth = float(depth_input)
            gap = float(gap_input)
        except ValueError:
            st.error("⚠️ Please enter valid numbers for significance, depth, and gap.")
            st.stop()

        # --- Zero check (sig & depth cannot be zero, gap can be zero) ---
        if sig == 0 and depth == 0:
            st.error("⚠️ Invalid input: Significance and depth cannot both be zero.")
            st.stop()
        elif sig == 0:
            st.error("⚠️ Invalid input: Significance cannot be zero.")
            st.stop()
        elif depth == 0:
            st.error("⚠️ Invalid input: Depth cannot be zero.")
            st.stop()

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

        # --- Gap check (0 is valid here) ---
        if gap <= 90:
            gap_status = "Highly reliable location (very significant)"
        elif gap <= 180:
            gap_status = "Reliable location (significant)"
        elif gap <= 270:
            gap_status = "Low reliability (less significant)"
        else:
            gap_status = "Unreliable location estimate (low significance)"

        # --- Prediction ---
        if "Unpredictable" not in sig_status and "Unpredictable" not in depth_status and "Unreliable" not in gap_status:
            try:
                input_array = np.array([[sig, depth, gap, latitude, longitude]])
                predicted_magnitude = model.predict(input_array)
                pred_value = float(predicted_magnitude.flatten()[0])

                st.subheader("🔍 Earthquake Prediction Result")
                st.write(f"**Significance:** {sig} → {sig_status}")
                st.write(f"**Depth:** {depth} km → {depth_status}")
                st.write(f"**Gap:** {gap}° → {gap_status}")
                st.write(f"**Location:** ({latitude}, {longitude}), {get_country(latitude, longitude)}")
                st.success(f"**Predicted Magnitude:** {pred_value:.2f}")
            except ValueError as e:
                st.error(f"Model input error: {e}")
        else:
            st.warning("Prediction skipped due to out-of-range values.")
else:
    st.info("Click anywhere on the map to select a location.")
