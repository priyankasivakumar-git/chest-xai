import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import os

from image_data_loader import train_ds, val_ds, test_ds

# -----------------------------
# LOAD BASE MODEL
# -----------------------------
base_model = tf.keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

# Freeze base model
base_model.trainable = False

# -----------------------------
# CUSTOM HEAD
# -----------------------------
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)

output = tf.keras.layers.Dense(5, activation='softmax')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)

# -----------------------------
# COMPILE
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# TRAIN
# -----------------------------
model.fit(train_ds, validation_data=val_ds, epochs=10)

# =============================
# ✅ FIXED EVALUATION
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

print("\n===== MobileNetV2 METRICS =====")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# -----------------------------
# SAVE FULL REPORT
# -----------------------------
full_report = classification_report(y_true_labels, y_pred_labels)

with open("mobilenet_results.txt", "w") as f:
    f.write(full_report)

# -----------------------------
# SAVE CSV (COMPARISON)
# -----------------------------
result = {
    "Model": "MobileNetV2",
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
model.save("mobilenetv2.h5")

print("\n✅ MobileNetV2 model saved")
print("✅ Results added to comparison.csv")