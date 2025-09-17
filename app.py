import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import math

# Load your trained model
@st.cache_resource
def load_model():
    # You'll need to save your model first: model.save('solar_model.h5')
    return tf.keras.models.load_model('solar_model.h5')

model = load_model()

st.title("🌞 Solar Power Prediction System")
st.write("Predict AC power output based on weather and time conditions")

# Create input form
col1, col2 = st.columns(2)

with col1:
    st.subheader("Weather Conditions")
    ambient_temp = st.slider("Ambient Temperature (°C)", 15, 40, 25)
    module_temp = st.slider("Module Temperature (°C)", 15, 70, 30)
    irradiation = st.slider("Solar Irradiation", 0.0, 1.2, 0.5, 0.01)

with col2:
    st.subheader("Time Settings")
    hour = st.slider("Hour of Day", 0, 23, 12)
    month = st.selectbox("Month", 
                        ["January", "February", "March", "April", "May", "June",
                         "July", "August", "September", "October", "November", "December"])
    
    # Convert month to number
    month_num = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
                 "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12}[month]

# Create cyclic features (same as training)
def create_cyclic_features(hour, month):
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)
    month_sin = math.sin(2 * math.pi * month / 12)
    month_cos = math.cos(2 * math.pi * month / 12)
    return hour_sin, hour_cos, month_sin, month_cos

# Make prediction
if st.button("🔮 Predict Solar Power", type="primary"):
    # Prepare input data
    hour_sin, hour_cos, month_sin, month_cos = create_cyclic_features(hour, month_num)
    
    input_data = np.array([[ambient_temp, module_temp, irradiation, 
                           hour_sin, hour_cos, month_sin, month_cos]])
    
    # Make prediction (assuming log-transformed model)
    prediction_log = model.predict(input_data, verbose=0)
    prediction = np.expm1(prediction_log[0][0])  # Transform back from log
    prediction = max(0, prediction)  # Ensure non-negative
    
    # Display results
    st.success(f"🔋 Predicted AC Power: **{prediction:.2f} W**")
    
    # Add context
    if prediction < 50:
        st.info("💡 Low power output - likely nighttime or poor weather conditions")
    elif prediction < 300:
        st.info("🌤️ Moderate power output - partial sunlight conditions")
    else:
        st.info("☀️ High power output - excellent solar conditions!")
    
    # Show input summary
    with st.expander("Input Summary"):
        st.write(f"🌡️ Ambient Temperature: {ambient_temp}°C")
        st.write(f"🌡️ Module Temperature: {module_temp}°C")
        st.write(f"☀️ Solar Irradiation: {irradiation}")
        st.write(f"🕐 Time: {hour:02d}:00 in {month}")

# Add model info
st.sidebar.markdown("## 📊 Model Information")
st.sidebar.write("- **Type**: Neural Network")
st.sidebar.write("- **Features**: Weather + Time")
st.sidebar.write("- **Output**: AC Power (Watts)")

# Instructions for running
st.sidebar.markdown("## 🚀 How to Run")
st.sidebar.code("streamlit run streamlit_app.py", language="bash")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit & TensorFlow")
