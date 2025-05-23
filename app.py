from flask import Flask, render_template, request, jsonify
from flask_ngrok import run_with_ngrok
from tensorflow import keras
from pyngrok import ngrok
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import gdown

# Model URL and local path
MODEL_URL = "https://drive.google.com/uc?id=1mWodSTMr2JbLAuMzfG9ChIxSIw96Mxwp"
MODEL_PATH = "Brain_Tumor_Detect.h5"

# Download the model if it does not exist locally
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
else:
    print("Model already exists locally.")

app = Flask(__name__)
run_with_ngrok(app)  # Start ngrok when app runs

# Ngrok auth token (replace with your token)
ngrok.set_auth_token("2saP5iSwMDpR9VmiyczwBjHRP5K_67inxBiL9DD6672553Et3")

# Open ngrok tunnel on port 5000
public_url = ngrok.connect(5000).public_url
print(f"Public URL: {public_url}")

# Load the keras model from local file
model = keras.models.load_model(MODEL_PATH)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Make sure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Process the image
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    image = np.repeat(image, 3, axis=-1)
    image = image.astype('float32')

    # Predict with the model
    prediction = model.predict(image)
    class_index = np.argmax(prediction)

    labels = ['pituitary', 'meningioma', 'glioma', 'notumor']
    result = labels[class_index]

    return jsonify({'prediction': result})

if __name__ == "__main__":
    app.run()

