from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

from tnbc3 import predict_breed


app = Flask(__name__)

CORS(app)


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:

        return jsonify({
            "error": "No image uploaded"
        }), 400


    file = request.files["image"]


    if file.filename == "":

        return jsonify({
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
        debug=True,
        port=5000
    )