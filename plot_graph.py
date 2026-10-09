import json
import matplotlib.pyplot as plt

# Load history
with open("history.json", "r") as f:
    history = json.load(f)

# -----------------------------
# ACCURACY GRAPH
# -----------------------------
plt.figure(figsize=(6,4))

plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')

plt.title('Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

plt.savefig("accuracy_graph.png", dpi=300)
plt.show()

# -----------------------------
# LOSS GRAPH
# -----------------------------
plt.figure(figsize=(6,4))

plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')

plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.savefig("loss_graph.png", dpi=300)
plt.show()