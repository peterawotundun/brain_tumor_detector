import streamlit as st
import numpy as np
from tensorflow import keras
import os
import gdown
from PIL import Image

# New Model URL and local path
MODEL_URL = "https://drive.google.com/uc?id=1YpQTwqlHaLXnrzTwKDVqoLRGZgchTKWp"
MODEL_PATH = "Brain_Tumor_Detect.h5"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load model
model = keras.models.load_model(MODEL_PATH)

# Labels
labels = ['pituitary', 'meningioma', 'glioma', 'notumor']

# App title
st.title("🧠 Brain Tumor Classifier")
st.write("Upload an MRI image to detect the type of brain tumor.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Prediction
if uploaded_file is not None:
    # Show image
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    image = np.repeat(image, 3, axis=-1).astype('float32')

    # Predict
    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    result = labels[class_index]

    # Display result
    st.success(f"*Prediction:* {result.capitalize()}")
