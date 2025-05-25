import tempfile

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files['image']

    # Use tempfile to avoid filename collisions
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        temp_path = temp_file.name

    try:
        result = CLIENT.infer(temp_path, model_id="my-first-project-tm3fw/1")
        os.remove(temp_path)

        # Extract the top predicted class string
        predicted_classes = result.get("predicted_classes", [])
        top_prediction = predicted_classes[0] if predicted_classes else "unknown"

        return jsonify({"prediction": top_prediction})

    except Exception as e:
        os.remove(temp_path)
        return jsonify({"error": str(e)}), 500
