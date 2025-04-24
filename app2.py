import os
import streamlit as st
import joblib
import numpy as np
import cv2
import demo  # Importing the demo.py file for Groq API check

# Disable oneDNN optimizations to avoid TensorFlow warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load the trained Random Forest model
rf_model = joblib.load("random_forest_model2.pkl")  

# Function to predict stroke
def predict_image(image):
    # Preprocess the image
    img = cv2.resize(image, (224, 224)).flatten()  # Resize & Flatten
    img = img.reshape(1, -1)  # Reshape for Random Forest input

    # Predict using Random Forest
    prediction = rf_model.predict(img)[0]

    return "\U0001F6D1 Stroke Detected!" if prediction == 1 else "\u2705 No Stroke Found"

# Streamlit Web App
st.title("\U0001F9E0 Brain Stroke Prediction usingRandom Forest Algorithm\n")
st.write("Upload a **Brain CT image** to check for **Stroke** or **Normal** Image.")

# Upload image
uploaded_file = st.file_uploader("\U0001F4C2 Choose an image...", type=["jpg", "png", "jpeg"])

is_brain = False  # Default value to prevent errors

if uploaded_file:
    try:
        # Read file as bytes
        file_bytes = uploaded_file.read()
        is_brain = demo.main(file_bytes)  # Pass raw bytes to demo.py
    except Exception as e:
        st.error(f"\u274C Error checking image with Groq API: {e}")

# Process only if a valid brain CT image is detected
if uploaded_file and is_brain:
    # Convert file bytes back to image
    image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("\U0001F50D Predict Stroke"):
        result = predict_image(image)
        st.success(f"### {result}")

elif uploaded_file and not is_brain:
    st.warning("\u26A0\uFE0F Please upload a correct Brain CT image.")
