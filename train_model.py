# =============================================================================
# --- PART 1: Data Setup, Configuration, and Local Data Transfer (Cell 1) ---
# =============================================================================

import os
import shutil
import time
import numpy as np
import tensorflow as tf
from google.colab import drive
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =============================
# 0. CLEAN MOUNT DIR + MOUNT DRIVE
# =============================

print("--- 1. Cleaning mount directory ---")
if os.path.exists('/content/drive'):
    shutil.rmtree('/content/drive')

os.makedirs('/content/drive', exist_ok=True)

print("--- 2. Mounting Google Drive ---")
drive.mount('/content/drive', force_remount=True)


# =============================
# 1. CONFIGURATION
# =============================

drive_dataset_root = "/content/drive/MyDrive/PlantVillage"   # Your dataset in Drive
local_dataset_path = "/content/plant_village_local"           # Fast training copy

NUM_CLASSES = 15
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
VALIDATION_SPLIT_RATIO = 0.2
MAX_EPOCHS = 50

MODEL_FILE_H5 = "plant_disease_model_15_class.h5"
HISTORY_FILE_PKL = "training_history_15_class.pkl"


# =============================
# 2. COPY DATA FROM DRIVE → LOCAL DISK
# =============================

print("\n--- 3. Copying dataset to fast local disk (this speeds training) ---")

# Remove old version if exists
if os.path.exists(local_dataset_path):
    shutil.rmtree(local_dataset_path)
    print(f"Removed old folder: {local_dataset_path}")

try:
    shutil.copytree(drive_dataset_root, local_dataset_path)
    print(f"Copied dataset successfully to: {local_dataset_path}")
except Exception as e:
    print("\n❌ ERROR: Cannot copy dataset from Drive.")
    print("Check path:", drive_dataset_root)
    print("Error details:", e)


# =============================
# 3. INITIALIZE DATA GENERATORS
# =============================

print("\n--- 4. Initializing Data Generators ---")

processing_parameters = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=VALIDATION_SPLIT_RATIO
)

# Training generator
new_training_images = processing_parameters.flow_from_directory(
    local_dataset_path,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    seed=42
)

# Validation generator
new_testing_images = processing_parameters.flow_from_directory(
    local_dataset_path,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    seed=42
)

TRAIN_STEPS = new_training_images.samples // BATCH_SIZE
VALIDATION_STEPS = new_testing_images.samples // BATCH_SIZE

print("\n--- DATASET READY ---")
print(f"Training Images:   {new_training_images.samples}")
print(f"Validation Images: {new_testing_images.samples}")
print("\n✅ Now run Cell 2 for model training.")



# =============================================================================
# --- PART 2: Model Definition, Training, Graph Generation, Saving (Cell 2) ---
# =============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve, auc,
    precision_recall_curve
)
from sklearn.preprocessing import label_binarize


# =============================
# 1. MODEL ARCHITECTURE
# =============================

print("\n--- 4. Building and Compiling CNN Model ---")

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),

    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

early_stopper = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)


# =============================
# 2. TRAIN THE MODEL
# =============================

print(f"\n--- 5. Starting Model Training (Max Epochs: {MAX_EPOCHS}) ---")
print(f"Total Training Images: {new_training_images.samples}")

start_time = time.time()

history = model.fit(
    new_training_images,
    steps_per_epoch=TRAIN_STEPS,
    epochs=MAX_EPOCHS,
    validation_data=new_testing_images,
    validation_steps=VALIDATION_STEPS,
    callbacks=[early_stopper],
    verbose=2
)

end_time = time.time()
training_time = end_time - start_time

print(f"\nTraining Time: {training_time/60:.2f} minutes")


# =============================
# 3. EVALUATE
# =============================

val_loss, val_acc = model.evaluate(new_testing_images, steps=VALIDATION_STEPS)
print(f"\nValidation Accuracy: {val_acc*100:.2f}%")
print(f"Validation Loss: {val_loss:.4f}")


# =============================
# 4. PREDICT ON VALIDATION SET
# =============================

print("\nGenerating predictions...")

y_true, y_pred, y_prob = [], [], []

for batch_x, batch_y in new_testing_images:
    p = model.predict(batch_x)
    y_prob.extend(p)
    y_pred.extend(np.argmax(p, axis=1))
    y_true.extend(np.argmax(batch_y, axis=1))

    if len(y_true) >= new_testing_images.samples:
        break

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

class_names = list(new_testing_images.class_indices.keys())
y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))


# =============================
# 5. GRAPH SAVE UTIL
# =============================

graphs_dir = os.path.join(drive_dataset_root, "graphs")
os.makedirs(graphs_dir, exist_ok=True)

def save_graph(name):
    fpath = os.path.join(graphs_dir, f"{name}.png")
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    print(f"Saved → {fpath}")


# =============================
# 6. GRAPH 1 — LEARNING RATE CURVE
# =============================

print("\n--- Plot 1: Learning Rate vs Epoch ---")

lrs = [model.optimizer.learning_rate.numpy()] * len(history.history['loss'])

plt.figure(figsize=(10,5))
plt.plot(lrs)
plt.title("Learning Rate per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.grid(True)
save_graph("learning_rate")
plt.show()


# =============================
# 7. GRAPH 2 — CONFUSION MATRIX
# =============================

print("\n--- Plot 2: Confusion Matrix ---")

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

plt.figure(figsize=(8,6))
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.grid(False)
save_graph("confusion_matrix")
plt.show()


# =============================
# 8. GRAPH 3 — CLASSIFICATION REPORT HEATMAP
# =============================

print("\n--- Plot 3: Classification Report Heatmap ---")

report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
matrix = np.array([list(report[c].values())[:-1] for c in class_names])

plt.figure(figsize=(10,6))
sns.heatmap(
    matrix,
    annot=True,
    cmap="YlGnBu",
    xticklabels=["Precision", "Recall", "F1-Score", "Support"],
    yticklabels=class_names
)
plt.title("Classification Report Heatmap")
save_graph("classification_report_heatmap")
plt.show()


# =============================
# 9. GRAPH 4 — ROC CURVE
# =============================

print("\n--- Plot 4: ROC Curve ---")

plt.figure(figsize=(10,7))

for i in range(NUM_CLASSES):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.2f})")

plt.plot([0,1], [0,1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (One-vs-Rest)")
plt.legend()
plt.grid(True)
save_graph("roc_curve")
plt.show()


# =============================
# 10. GRAPH 5 — PRECISION-RECALL CURVE
# =============================

print("\n--- Plot 5: Precision–Recall Curve ---")

plt.figure(figsize=(10,7))

for i in range(NUM_CLASSES):
    prec, rec, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
    plt.plot(rec, prec, label=class_names[i])

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.legend()
plt.grid(True)
save_graph("precision_recall_curve")
plt.show()


# =============================
# 11. SAVE MODEL + HISTORY
# =============================

print("\n--- Saving Model & Training History ---")

model.save(os.path.join(drive_dataset_root, MODEL_FILE_H5))
print("Model Saved ✔")

with open(os.path.join(drive_dataset_root, HISTORY_FILE_PKL), "wb") as f:
    pickle.dump(history.history, f)

print("Training History Saved ✔")

print("\n🔥 DONE: Model, history, and all 5 graphs saved successfully.")
