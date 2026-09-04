from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

from tnbc3 import predict_breed

app = Flask(__name__)

# Allow requests from your Vercel frontend
CORS(app)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "Cattle Breed AI API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image uploaded"
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "success": False,
            "error": "No image selected"
        }), 400

    try:
        image = Image.open(file.stream)
        predictions = predict_breed(image)

        return jsonify({
            "success": True,
            "predictions": predictions
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )