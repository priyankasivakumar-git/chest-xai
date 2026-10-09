import tensorflow as tf
import json

# -----------------------------
# MIXED PRECISION (SET FIRST)
# -----------------------------
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# -----------------------------
# GPU MEMORY FIX
# -----------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# -----------------------------
# IMPORT DATA & MODEL
# -----------------------------
from data_loader import train_ds, val_ds
from model import model

# -----------------------------
# CLASS WEIGHTS
# -----------------------------
class_weights = {
    0: 0.7980,
    1: 0.5579,
    2: 2.1189,
    3: 0.7849,
    4: 4.7914
}

# -----------------------------
# CALLBACKS
# -----------------------------
callbacks = [

    # Early stopping
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),

    # Save best model
    tf.keras.callbacks.ModelCheckpoint(
        "best_model.h5",
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),

    # Reduce LR
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

# -----------------------------
# TRAIN MODEL
# -----------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    class_weight=class_weights,
    callbacks=callbacks
)

# -----------------------------
# SAVE FINAL MODEL
# -----------------------------
model.save("final_model.h5")

# -----------------------------
# ✅ FIXED HISTORY SAVE
# -----------------------------
history_dict = {}

for key, values in history.history.items():
    history_dict[key] = [float(v) for v in values]

with open("history.json", "w") as f:
    json.dump(history_dict, f)

# -----------------------------
# DONE
# -----------------------------
print("✅ Training completed!")
print("✅ Model saved!")
print("✅ History saved!")