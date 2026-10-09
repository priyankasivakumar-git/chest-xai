import tensorflow as tf

IMG_SIZE = 224
BATCH_SIZE = 32
DATASET_PATH = "dataset"

# -----------------------------
# TRAIN DATA
# -----------------------------
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# ✅ GET CLASS NAMES HERE
class_names = train_ds.class_names
print("Classes:", class_names)

# -----------------------------
# VALIDATION DATA
# -----------------------------
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# -----------------------------
# TEST SPLIT
# -----------------------------
val_batches = tf.data.experimental.cardinality(val_ds)

test_ds = val_ds.take(val_batches // 2)
val_ds = val_ds.skip(val_batches // 2)

# -----------------------------
# PERFORMANCE OPTIMIZATION
# -----------------------------
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)