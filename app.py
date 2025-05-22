from flask import Flask, request, jsonify, send_from_directory
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import requests

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

interpreter = tf.lite.Interpreter(model_path="pill_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

def prepare_image(image_path):
    img = Image.open(image_path).resize((224, 224))
    img = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img, axis=0)

