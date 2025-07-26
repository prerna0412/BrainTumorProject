
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the model
model = load_model("brain_tumor_transfer_model.h5")

# Define class names
class_names = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]

# Streamlit UI
st.set_page_config(page_title="🧠 Brain Tumor Classifier", layout="centered")
st.title("🎯 Brain Tumor MRI Classification")
st.markdown("Upload a brain MRI image and let the AI predict the tumor type.")

uploaded_file = st.file_uploader("📁 Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded MRI", use_column_width=True)

    # Preprocess the image
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"🧠 Prediction: **{predicted_class}**")
    st.info(f"📊 Confidence: **{confidence:.2f}%**")
