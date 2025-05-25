from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from inference_sdk import InferenceHTTPClient
import os

app = Flask(__name__)
CORS(app)

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="PPOn3zoc59OqaXFYyDrZ"
)

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

        predictions = result.get("predictions", [])
        if len(predictions) > 0:
            top_pred = predictions[0]
            predicted_class = top_pred.get("class", "Unknown")
            confidence = top_pred.get("confidence", 0)

            return jsonify({
                "prediction": predicted_class,
                "confidence": confidence
            })
        else:
            return jsonify({"error": "No predictions found"}), 500

    except Exception as e:
        os.remove(file_path)
        return jsonify({"error": str(e)}), 500
