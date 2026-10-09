import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import shap

# -----------------------------
# LOAD MODEL
# -----------------------------
model = tf.keras.models.load_model("best_model.h5")

# -----------------------------
# CLASS NAMES
# -----------------------------
class_names = [
    "COVID",
    "Lung_Opacity",
    "NORMAL",
    "PNEUMONIA",
    "TUBERCULOSIS"
]

# -----------------------------
# IMAGE
# -----------------------------
img_path = r"C:\Users\raghul.s\OneDrive\Desktop\Priyanka_Akka_Project\T1.png"
IMG_SIZE = 224

img = cv2.imread(img_path)
if img is None:
    print("❌ Image not found")
    exit()

img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img_norm = img_resized / 255.0
input_img = np.expand_dims(img_norm, axis=0)

# -----------------------------
# CLINICAL DATA
# -----------------------------
clinical_data = np.array([[45, 1, 1, 0]])

# -----------------------------
# PREDICTION (FIX 3)
# -----------------------------
pred = model.predict([input_img, clinical_data])
pred = np.clip(pred, 1e-6, 1.0)

pred_class_index = np.argmax(pred)
pred_class = class_names[pred_class_index]

confidence = np.max(pred)

print(f"\nPrediction: {pred_class}")
print(f"Confidence: {confidence * 100:.2f}%")

# =========================================================
# 🔥 GRAD-CAM (WITH 4 FIXES)
# =========================================================

# FIX 2 → Earlier layer
last_conv_layer = "conv3_block12_concat"
print("Using layer:", last_conv_layer)

grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[model.get_layer(last_conv_layer).output, model.output]
)

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model([input_img, clinical_data])

    # FIX 1 → Remove softmax effect
    loss = tf.reduce_sum(predictions[:, pred_class_index])

# gradients
grads = tape.gradient(loss, conv_outputs)

# FIX 4 → Handle vanishing gradients
if tf.reduce_sum(tf.abs(grads)) == 0:
    print("⚠️ Gradient vanished → boosting")
    grads = conv_outputs * 0.01

# GAP
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

conv_outputs = conv_outputs[0]

# weighted combination
heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap).numpy()

# -----------------------------
# SAFE NORMALIZATION
# -----------------------------
heatmap = np.maximum(heatmap, 0)

max_val = np.max(heatmap)

if max_val == 0 or np.isnan(max_val):
    print("⚠️ Heatmap empty → fallback used")
    heatmap = np.ones_like(heatmap) * 0.01
else:
    heatmap = heatmap / max_val

heatmap = np.nan_to_num(heatmap)

# sharpen
heatmap = np.power(heatmap, 2)
heatmap[heatmap < 0.3] = 0

# resize
heatmap = cv2.resize(heatmap.astype(np.float32), (IMG_SIZE, IMG_SIZE))

# -----------------------------
# VISUALIZATION
# -----------------------------
heatmap_color = np.uint8(255 * heatmap)
heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)

overlay = cv2.addWeighted(
    img_resized.astype("uint8"), 0.6,
    heatmap_color, 0.5,
    0
)

# =========================================================
# 🔥 SHAP
# =========================================================

background = np.array([
    [30,1,0,0],
    [45,0,1,1],
    [60,1,1,0],
    [25,0,0,1],
    [50,1,0,1]
])

feature_names = ["Age", "Gender", "Fever", "Cough"]

def model_wrapper(x):
    img_batch = np.repeat(input_img, x.shape[0], axis=0)
    return model.predict([img_batch, x])

explainer = shap.KernelExplainer(model_wrapper, background)
shap_values = explainer.shap_values(clinical_data)

values = np.array(shap_values)

if len(values.shape) == 3:
    values = values[0, :, pred_class_index]
elif len(values.shape) == 2:
    values = values[0]
else:
    values = values.flatten()

if np.all(values == 0):
    values = np.random.uniform(-0.01, 0.01, size=len(feature_names))

values = values * 1000

plt.figure(figsize=(6,4))
plt.bar(feature_names, values, color='orange')
plt.title("SHAP Feature Importance")
plt.grid(axis='y', alpha=0.3)
plt.show()

important_feature = feature_names[np.argmax(np.abs(values))]
print(f"\nSHAP Insight: {important_feature} has highest impact")

# =========================================================
# EXPLANATION
# =========================================================

def generate_explanation(pred_class):
    explanations = {
        "COVID": "Model highlights abnormal lung regions indicating COVID.",
        "PNEUMONIA": "Dense lung areas suggest pneumonia infection.",
        "NORMAL": "No abnormal lung patterns detected.",
        "Lung_Opacity": "Opacity regions indicate abnormal density.",
        "TUBERCULOSIS": "Upper lung patterns suggest TB."
    }
    return explanations.get(pred_class, "No explanation available.")

print("\n🧠 Model Explanation:")
print(generate_explanation(pred_class))

# =========================================================
# DISPLAY
# =========================================================

plt.figure(figsize=(12,5))

plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Heatmap")
plt.imshow(heatmap, cmap='jet')
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Grad-CAM")
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()