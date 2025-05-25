from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from inference_sdk import InferenceHTTPClient
import os

app = Flask(__name__)
CORS(app)

# ✅ Initialize once globally
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="PPOn3zoc59OqaXFYyDrZ"
)

@app.route("/")
def home():
    return send_file("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files['image']
    file_path = "temp_" + image.filename
    image.save(file_path)

    try:
        result = CLIENT.infer(file_path, model_id="my-first-project-tm3fw/1")
        os.remove(file_path)

        predicted_classes = result.get("predicted_classes", [])
        top_prediction = predicted_classes[0] if predicted_classes else "unknown"

        return jsonify({"prediction": top_prediction})

    except Exception as e:
        os.remove(file_path)
        return jsonify({"error": str(e)}), 500
