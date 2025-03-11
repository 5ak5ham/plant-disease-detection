from flask import Flask, request, jsonify
from tensorflow import keras
import tensorflow as tf
import numpy as np
import os
import cv2
import gdown

app = Flask(__name__)

# MODEL_URL = "https://drive.google.com/uc?id=154aOnfcvEA47Um-RLXprFNal3g-QvVEh"
# MODEL_PATH = "/opt/render/persistent/trained_model.h5"
MODEL_PATH = "trained_model.h5"
# def download_model():
#     os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)  # Ensure directory exists
#     if not os.path.exists(MODEL_PATH):
#         gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# download_model()
# Load the model
model = keras.models.load_model(MODEL_PATH)

# Preprocess image - using the exact same preprocessing as your test code
def preprocess_image(image_path):
    # Use tf.keras.preprocessing.image functions to match your testing code
    image = tf.keras.utils.load_img(image_path, target_size=(128, 128))
    input_arr = tf.keras.utils.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    return input_arr

# Define home route
@app.route('/')
def home():
    return "Welcome to the Plant Disease Detection API! Use the /predict endpoint to classify plant diseases."

# Define route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    # Save the uploaded file temporarily
    temp_path = "temp_upload.jpg"
    file.save(temp_path)
    
    # Process the image using the same approach as your testing code
    input_arr = preprocess_image(temp_path)
    
    # Make prediction
    predictions = model.predict(input_arr)
    class_index = np.argmax(predictions[0])  # Get highest probability class
    
    # Get class names from your training directory
    class_names = [
    "Apple Scab", "Apple Black Rot", "Apple Cedar Apple Rust", "Apple Healthy",
    "Blueberry Healthy", "Cherry (Including Sour) Healthy", "Cherry (Including Sour) Powdery Mildew",
    "Corn (Maize) Cercospora Leaf Spot Gray Leaf Spot", "Corn (Maize) Common Rust",
    "Corn (Maize) Healthy", "Corn (Maize) Northern Leaf Blight", 
    "Grape Black Rot", "Grape Esca (Black Measles)", "Grape Healthy",
    "Grape Leaf Blight (Isariopsis Leaf Spot)", "Orange Haunglongbing (Citrus Greening)",
    "Peach Bacterial Spot", "Peach Healthy", "Pepper Bell Bacterial Spot", "Pepper Bell Healthy",
    "Potato Early Blight", "Potato Healthy", "Potato Late Blight", 
    "Raspberry Healthy", "Soybean Healthy", "Squash Powdery Mildew",
    "Strawberry Healthy", "Strawberry Leaf Scorch",
    "Tomato Bacterial Spot", "Tomato Early Blight", "Tomato Healthy",
    "Tomato Late Blight", "Tomato Leaf Mold", "Tomato Septoria Leaf Spot",
    "Tomato Spider Mites Two-Spotted Spider Mite", "Tomato Target Spot",
    "Tomato Mosaic Virus", "Tomato Yellow Leaf Curl Virus"
    ]
    
    # Create result
    result = {
        "prediction": class_names[class_index],
        "confidence": float(predictions[0][class_index])  # Convert numpy float to Python float
    }
    
    # Clean up the temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)
        
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
