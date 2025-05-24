import streamlit as st
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
import gdown

# URLs & Paths
MODEL_URL = "https://drive.google.com/uc?id=1mWodSTMr2JbLAuMzfG9ChIxSIw96Mxwp"
MODEL_PATH = "Brain_Tumor_Detect.h5"

# Download model if not present
@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

model = download_model()

labels = ['pituitary', 'meningioma', 'glioma', 'notumor']

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("🧠 Brain Tumor Detection App")
st.write("Upload an MRI image (JPG/PNG) to detect if there's a tumor.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if image is not None:
        st.image(image, caption="Uploaded Image", use_column_width=True)

        image_resized = cv2.resize(image, (224, 224))
        image_resized = image_resized / 255.0
        image_resized = np.expand_dims(image_resized, axis=(0, -1))  # (1, 224, 224, 1)
        image_resized = np.repeat(image_resized, 3, axis=-1)  # (1, 224, 224, 3)

        if st.button("🔍 Predict"):
            with st.spinner("Running prediction..."):
                prediction = model.predict(image_resized)
                class_index = int(np.argmax(prediction))
                confidence = float(np.max(prediction))
                result = labels[class_index]

            st.success(f"🧠 Prediction: {result.upper()}")
            st.info(f"Confidence: {confidence:.2%}")
    else:
        st.error("Could not read the image. Try another file.")
