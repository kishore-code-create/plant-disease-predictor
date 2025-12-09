# -------------------------------------------------------
# IMPORTS
# -------------------------------------------------------
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from google.colab import files

# -------------------------------------------------------
# 1. UPLOAD IMAGE FROM PC
# -------------------------------------------------------
uploaded = files.upload()  # Prompts you to select a file
IMAGE_PATH = list(uploaded.keys())[0]
print(f"Uploaded image: {IMAGE_PATH}")

# -------------------------------------------------------
# 2. SET MODEL & DATASET PATHS
# -------------------------------------------------------
MODEL_PATH = "/content/drive/MyDrive/PlantVillage/plant_disease_model_15_class.h5"
DATASET_DIR = "/content/drive/MyDrive/PlantVillage"
IMAGE_SIZE = (150, 150)  # Must match your model input

# -------------------------------------------------------
# 3. LOAD MODEL
# -------------------------------------------------------
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# -------------------------------------------------------
# 4. LOAD CLASS NAMES
# -------------------------------------------------------
class_names = sorted([
    folder for folder in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, folder)) and folder != "graphs"
])

print("\nDetected Classes:")
for i, c in enumerate(class_names):
    print(f"{i}: {c}")

# -------------------------------------------------------
# 5. PREPROCESS FUNCTION
# -------------------------------------------------------
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, IMAGE_SIZE)
    img_normalized = img_resized / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)
    return img, img_expanded

# -------------------------------------------------------
# 6. MAKE PREDICTION
# -------------------------------------------------------
orig_img, input_img = preprocess_image(IMAGE_PATH)

pred = model.predict(input_img)[0]
top1 = np.argmax(pred)
top3 = pred.argsort()[-3:][::-1]

# -------------------------------------------------------
# 7. DISPLAY RESULT
# -------------------------------------------------------
plt.imshow(orig_img)
plt.axis("off")
plt.title("Input Image")
plt.show()

print("\n🔥 TOP 1 PREDICTION 🔥")
print(f"Class: {class_names[top1]}")
print(f"Confidence: {pred[top1] * 100:.2f}%\n")

print("🔎 TOP 3 Predictions:")
for idx in top3:
    print(f"{class_names[idx]} --> {pred[idx] * 100:.2f}%")
