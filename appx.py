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

# Load the model
model = keras.models.load_model(MODEL_PATH)

# Label mapping
labels = ['pituitary', 'meningioma', 'glioma', 'notumor']

# Streamlit UI
st.title("🧠 Brain Tumor Detection")
st.write("Upload a brain MRI scan image to detect the type of tumor.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # Preprocess the image
    image_resized = cv2.resize(image, (224, 224))
    image_resized = image_resized / 255.0
    image_resized = np.expand_dims(image_resized, axis=0)
    image_resized = np.expand_dims(image_resized, axis=-1)
    image_resized = np.repeat(image_resized, 3, axis=-1)
    image_resized = image_resized.astype('float32')

    # Predict
    if st.button("Predict"):
        prediction = model.predict(image_resized)
        class_index = np.argmax(prediction)
        result = labels[class_index]
        confidence = np.max(prediction)

        st.success(f"Prediction: **{result.upper()}**")
        st.write(f"Confidence: {confidence:.2%}")
