@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files['image']
    file_path = "temp_" + image.filename
    image.save(file_path)

    try:
        # Initialize client here (lightweight + safe)
        CLIENT = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key="PPOn3zoc59OqaXFYyDrZ"
        )

        result = CLIENT.infer(file_path, model_id="my-first-project-tm3fw/1")
        os.remove(file_path)
        return jsonify(result)
    except Exception as e:
        os.remove(file_path)
        return jsonify({"error": str(e)}), 500
