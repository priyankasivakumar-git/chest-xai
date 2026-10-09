import tensorflow as tf
import numpy as np
import cv2

# -----------------------------
# LOAD MODEL
# -----------------------------
model = tf.keras.models.load_model("best_model.h5")

# -----------------------------
# CLASS NAMES (IMPORTANT)
# -----------------------------
class_names = [
    "COVID",
    "Lung_Opacity",
    "NORMAL",
    "PNEUMONIA",
    "TUBERCULOSIS"
]

# -----------------------------
# LOAD IMAGE
# -----------------------------
img_path = r"C:\Users\raghul.s\OneDrive\Desktop\Priyanka_Akka_Project\n1.jpeg"  # 🔥 change this

IMG_SIZE = 224

img = cv2.imread(img_path)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# -----------------------------
# DUMMY CLINICAL DATA
# -----------------------------
clinical_data = np.array([[45, 1, 1, 0]])  
# [age, gender, fever, cough]

# -----------------------------
# PREDICTION
# -----------------------------
pred = model.predict([img, clinical_data])

pred_class = np.argmax(pred)
confidence = np.max(pred)

# -----------------------------
# OUTPUT
# -----------------------------
print("Predicted Class:", class_names[pred_class])
print("Confidence:", round(confidence * 100, 2), "%")