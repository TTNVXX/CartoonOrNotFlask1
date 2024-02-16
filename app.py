import flask
from flask import Flask, request, jsonify
from flask_uploads import UploadSet, configure_uploads, IMAGES
from PIL import Image
import numpy as np
import tensorflow as tf
from transformers import AutoImageProcessor, SwinForImageClassification
import torch
import os

app = flask.Flask(__name__)

# Configure file uploads
app.config['UPLOADED_IMAGES_DEST'] = 'static/uploads'  # Ensure directory exists
MAX_FILE_SIZE = 1024 * 1024 * 5  # Set your desired file size limit in bytes (5 MB in this example)
ALLOWED_MIMETYPES = ['image/jpeg', 'image/png', 'image/gif']  # Add allowed file types
images = UploadSet('images', IMAGES)
configure_uploads(app, images)

image_processor = AutoImageProcessor.from_pretrained("Libidrave/CartoonOrNotv2")
model = SwinForImageClassification.from_pretrained("Libidrave/CartoonOrNotv2")

@app.route("/")
def hello():
    return "Cartoon or Not: Image Prediction"

@app.route("/predict_image", methods=["POST"])
def predict_image():
    try:
        # File handling and validation
        if 'file' not in request.files:
            return "No file found", 400

        image = request.files['file']
        if image.filename == '':
            return "No selected file", 400

        if image.content_length > MAX_FILE_SIZE:
            return "File size exceeds limit", 400

        # Validate file type using allowed MIME types
        if image.mimetype not in ALLOWED_MIMETYPES:
            return "Invalid file type", 400

        # Save the uploaded image
        filename = images.save(image)

        # Process and predict using saved image
        img_path = os.path.join(app.config['UPLOADED_IMAGES_DEST'], filename)
        image = Image.open(img_path)
        image = image.resize((224, 224))
        img_array = tf.keras.utils.img_to_array(image)

        inputs = image_processor(img_array, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_label = torch.argmax(logits, dim=1).item()
        probabilities = torch.softmax(logits, dim=1)

        # Return informative response with prediction accuracy and image URL
        return jsonify({
            "prediction": model.config.id2label[predicted_label],
            "accuracy": float(probabilities[0][predicted_label]) * 100
        })

    except Exception as e:
        # Log error and return appropriate response
        app.logger.error(f"Error predicting image: {str(e)}")
        return "Internal Server Error: An error occurred while processing the image", 500

if __name__ == "__main__":
    app.run(debug=True)
