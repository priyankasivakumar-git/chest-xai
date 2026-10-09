import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# -----------------------------
# IMAGE INPUT
# -----------------------------
image_input = Input(shape=(224, 224, 3))

base_model = DenseNet121(
    weights='imagenet',
    include_top=False,
    input_tensor=image_input
)

# Freeze layers
for layer in base_model.layers[-20:]:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)

# EMWF (image)
img_weight = Dense(1, activation='sigmoid')(x)
x = Multiply()([x, img_weight])

# -----------------------------
# CLINICAL INPUT
# -----------------------------
clinical_input = Input(shape=(4,))

c = Dense(64, activation='relu')(clinical_input)

# EMWF (clinical)
clin_weight = Dense(1, activation='sigmoid')(c)
c = Multiply()([c, clin_weight])

# -----------------------------
# FUSION
# -----------------------------
combined = concatenate([x, c])

z = Dense(128, activation='relu')(combined)
z = Dropout(0.5)(z)

output = Dense(5, activation='softmax')(z)

# -----------------------------
# MODEL
# -----------------------------
model = Model(inputs=[image_input, clinical_input], outputs=output)


optimizer = Adam(learning_rate=0.0001)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()