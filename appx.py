import streamlit as st
import cv2
import numpy as np
import os
from tensorflow import keras
import gdown

# Constants
MODEL_URL = "https://drive.google.com/uc?id=1mWodSTMr2JbLAuMzfG9ChIxSIw96Mxwp"
MODEL_PATH = "Brain_Tumor_Detect.h5"

# Download the model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
else:
    st.sidebar.success("✅ Model loaded from local storage.")

# Load the model
@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)

model = load_model()

# Labels
labels = ['pituitary', 'meningioma', 'glioma', 'notumor']

# Streamlit App
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("🧠 Brain Tumor Detection using MRI Scans")
st.write("Upload an MRI scan image (JPG/PNG) to detect the presence and type of brain tumor.")

# File Uploader
uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if image is None:
        st.error("Error: Could not read image. Please upload a valid image file.")
    else:
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocessing
        image_resized = cv2.resize(image, (224, 224))
        image_resized = image_resized / 255.0
        image_resized = np.expand_dims(image_resized, axis=0)
        image_resized = np.expand_dims(image_resized, axis=-1)
        image_resized = np.repeat(image_resized, 3, axis=-1)
        image_resized = image_resized.astype('float32')

        # Prediction
        if st.button("🔍 Predict Tumor Type"):
            with st.spinner("Analyzing..."):
                prediction = model.predict(image_resized)
                class_index = np.argmax(prediction)
                result = labels[class_index]
                confidence = float(np.max(prediction))

            st.success(f"🎯 Prediction: **{result.upper()}**")
            st.info(f"🧪 Confidence: **{confidence:.2%}**")
