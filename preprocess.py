from PIL import Image
import numpy as np
import tensorflow as tf
import torch
from transformers import AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained("Libidrave/CartoonOrNotv2")

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(image)
    inputs = image_processor(img_array, return_tensors="pt")
    return inputs

def predict(model, inputs):
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_label = torch.argmax(logits, dim=1).item()
    probabilities = torch.softmax(logits, dim=1)
    return predicted_label, probabilities
