# Import required packages
from flask import Flask, render_template, request, url_for
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os

app = Flask(__name__)

# Project introduction info (this will be embedded into the HTML)
intro_text = """
This tool was developed for a final project in Artificial Intelligence at Plymouth University, 
using an advanced AI model with 97% accuracy to predict chest conditions.
"""

# Load the AI model for prediction
model_path = "Chest_Xray_Model_Tf_2_15.h5"
model = load_model(model_path)

# Function to predict image class
def predict_image(image):
    image = image.convert('RGB')
    # Preprocess the image for prediction
    img = np.array(image)
    img = tf.image.resize(img, (64, 64))  # Resize to the expected input size
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img)
    class_index = np.argmax(prediction)

    # Determine the condition based on prediction
    if class_index == 0:
        label = "Covid-19"
    elif class_index == 1:
        label = "Normal"
    elif class_index == 2:
        label = "Viral Pneumonia"
    elif class_index == 3:
        label = "Tuberculosis"
    else:
        label = "Invalid Image"

    return label

# Define the route to handle image upload and prediction
@app.route("/", methods=["GET", "POST"])
def index():
    image = None
    label = None
    advice = None

    if request.method == "POST":
        uploaded_file = request.files.get("file")
        if uploaded_file:
            # Process the uploaded file
            image = Image.open(uploaded_file)
            image = image.convert("RGB")
            image_path = os.path.join("static", "uploaded_image.jpg")
            image.save(image_path)
            
            # Get the prediction label
            label = predict_image(image)

            # Provide advice based on the prediction
            if label == "Covid-19":
                advice = [
                    "Seek medical help immediately.",
                    "Isolate yourself to prevent spreading the virus to others.",
                    "Follow local health guidelines and inform your close contacts."
                ]
            elif label == "Normal":
                advice = [
                    "Your chest X-ray appears normal.",
                    "Maintain a healthy lifestyle and follow preventive measures.",
                    "If you have symptoms, consult a healthcare provider."
                ]
            elif label == "Viral Pneumonia":
                advice = [
                    "Consult a healthcare provider promptly.",
                    "Follow the treatment plan prescribed by your doctor.",
                    "Rest and stay hydrated for recovery."
                ]
            elif label == "Tuberculosis":
                advice = [
                    "Seek immediate medical attention.",
                    "Follow the prescribed treatment regimen strictly.",
                    "Inform your close contacts as tuberculosis is contagious."
                ]
            elif label == "Invalid Image":
                advice = [
                    "The image you uploaded does not appear to be a valid chest X-ray.",
                    "Please upload a clear chest X-ray image for proper diagnosis."
                ]
    
    return render_template("index.html", image=image, label=label, advice=advice, intro_text=intro_text)

if __name__ == "__main__":
    app.run(debug=True)

#create a static folder (empty) and templates folder containing index.html and result.html 