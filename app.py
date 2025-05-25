from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from inference_sdk import InferenceHTTPClient
import os

app = Flask(__name__)
CORS(app)

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="PPOn3zoc59OqaXFYyDrZ"
)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

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

        if 'predictions' in result and len(result['predictions']) > 0:
            top_prediction = result['predictions'][0]
            return jsonify({
                "class": top_prediction["class"],
                "confidence": round(top_prediction["confidence"] * 100, 2)
            })
        else:
            return jsonify({"message": "No tumor detected."})

    except Exception as e:
        os.remove(file_path)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
