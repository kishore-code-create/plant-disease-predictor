import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# -------------------------------
# Load model
# -------------------------------
MODEL_PATH = "plant_disease_model_15_class.h5"
model = tf.keras.models.load_model(MODEL_PATH)
st.sidebar.success("Model loaded successfully!")

# -------------------------------
# Class names
# -------------------------------
class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

# -------------------------------
# App UI
# -------------------------------
st.title("🌿 Plant Leaf Disease Predictor")
st.write("Upload a leaf image and the app will predict its disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    # Convert to numpy array
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess
    IMAGE_SIZE = (150, 150)
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img, IMAGE_SIZE)
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)
    
    # Predict
    pred = model.predict(img_input)[0]
    top1 = np.argmax(pred)
    top3 = pred.argsort()[-3:][::-1]
    
    # Display predictions
    st.write("## 🔥 Top 1 Prediction")
    st.write(f"Class: {class_names[top1]}")
    st.write(f"Confidence: {pred[top1]*100:.2f}%")
    
    st.write("## 🔎 Top 3 Predictions")
    for idx in top3:
        st.write(f"{class_names[idx]} --> {pred[idx]*100:.2f}%")
