import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import os

from image_data_loader import train_ds, val_ds, test_ds

# -----------------------------
# MODEL
# -----------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224,224,3)),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

# -----------------------------
# COMPILE
# -----------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# TRAIN
# -----------------------------
model.fit(train_ds, validation_data=val_ds, epochs=10)

# =============================
# ✅ FIXED EVALUATION (IMPORTANT)
# =============================

y_true_labels = []
y_pred_labels = []

for images, labels in test_ds:
    preds = model.predict(images)

    y_pred_labels.extend(np.argmax(preds, axis=1))
    y_true_labels.extend(np.argmax(labels.numpy(), axis=1))

y_true_labels = np.array(y_true_labels)
y_pred_labels = np.array(y_pred_labels)

# -----------------------------
# METRICS
# -----------------------------
report_dict = classification_report(y_true_labels, y_pred_labels, output_dict=True)

accuracy = accuracy_score(y_true_labels, y_pred_labels)
precision = report_dict['weighted avg']['precision']
recall = report_dict['weighted avg']['recall']
f1 = report_dict['weighted avg']['f1-score']

print("\n===== METRICS =====")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# -----------------------------
# SAVE FULL REPORT
# -----------------------------
full_report = classification_report(y_true_labels, y_pred_labels)

with open("custom_cnn_results.txt", "w") as f:
    f.write(full_report)

# -----------------------------
# SAVE CSV (COMPARISON)
# -----------------------------
result = {
    "Model": "Custom CNN",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1": f1
}

df = pd.DataFrame([result])

df.to_csv(
    "comparison.csv",
    mode='a',
    header=not os.path.exists("comparison.csv"),
    index=False
)

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save("custom_cnn.h5")

print("\n✅ Model saved")
print("✅ Results saved to comparison.csv")