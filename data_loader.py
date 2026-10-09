import pandas as pd
import tensorflow as tf
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


df = pd.read_csv("dataset_encoded.csv")


train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    stratify=df["label"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label"],
    random_state=42
)


IMG_SIZE = 224
BATCH_SIZE = 16


def create_dataset(df):

    paths = df["image_path"].values
    labels = df["label"].values
    clinical = df[["age", "gender", "fever", "cough"]].values

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels, clinical))

    def load_data(path, label, clinical):

        def _load_image(p):
            img = cv2.imread(p.decode())
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            return img.astype(np.float32)

        img = tf.numpy_function(_load_image, [path], tf.float32)
        img.set_shape((IMG_SIZE, IMG_SIZE, 3))

        return (img, clinical), tf.one_hot(label, 5)

    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


train_ds = create_dataset(train_df)
val_ds = create_dataset(val_df)


for (img, clin), label in train_ds.take(1):
    print("Image shape:", img.shape)
    print("Clinical shape:", clin.shape)
    print("Label shape:", label.shape)