from flask import Flask, request, jsonify
from PIL import Image
from tnbc3 import predict_breed

app = Flask(__name__)


@app.route("/api/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image uploaded"
        }), 400

    try:
        image = Image.open(
            request.files["image"]
        ).convert("RGB")

        predictions = predict_breed(image)

        return jsonify({
            "success": True,
            "predictions": predictions
        })

    except Exception as e:
        print(e)

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500