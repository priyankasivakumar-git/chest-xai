import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("comparison.csv")

# Add proposed model
proposed = pd.DataFrame([{
    "Model": "Proposed",
    "Accuracy": 0.98,
    "Precision": 0.98,
    "Recall": 0.98,
    "F1": 0.98
}])

df = pd.concat([df, proposed], ignore_index=True)

# -----------------------------
# BAR GRAPH (SAME COLOR)
# -----------------------------
plt.figure(figsize=(9,5))

bars = plt.bar(df["Model"], df["Accuracy"]*100, color='steelblue')

plt.title("Comparison of CNN Models")
plt.xlabel("Models")
plt.ylabel("Accuracy (%)")

plt.xticks(rotation=30)
plt.grid(axis='y')

# Value labels
for i, v in enumerate(df["Accuracy"]*100):
    plt.text(i, v + 1, f"{v:.1f}%", ha='center')

plt.show()