import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

# load model
model = tf.keras.models.load_model("best_model.h5")

# import dataset
from data_loader import val_ds   # or data_loader_split if using that

# -----------------------------
# GET TRUE & PRED LABELS
# -----------------------------
y_true = []
y_pred = []

for (img, clin), label in val_ds:
    preds = model.predict([img, clin], verbose=0)

    y_true.extend(np.argmax(label.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,6))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()

plt.xticks(range(5), range(5))
plt.yticks(range(5), range(5))

for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i, j], ha='center', va='center')

plt.show()

# -----------------------------
# CLASSIFICATION REPORT
# -----------------------------
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))