import streamlit as st
import numpy as np
import cv2
import os
import gdown
from tensorflow import keras

# Model download from Google Drive
MODEL_URL = "https://drive.google.com/uc?id=1mWodSTMr2JbLAuMzfG9ChIxSIw96Mxwp"
MODEL_PATH = "Brain_Tumor_Detect.h5"

# Download model if not available locally
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load the model
model = keras.models.load_model(MODEL_PATH)

# Title
st.title("🧠 Brain Tumor Detection")
st.markdown("Upload an MRI scan image to detect tumor type.")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    st.image(image, channels="GRAY", caption="Uploaded MRI", use_column_width=True)

    # Preprocess image
    img = cv2.resize(image, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    img = np.repeat(img, 3, axis=-1)
    img = img.astype("float32")

    # Predict
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    labels = ['pituitary', 'meningioma', 'glioma', 'notumor']
    result = labels[class_index]

    # Show result
    st.success(f"🧬 Prediction: **{result.upper()}**")
