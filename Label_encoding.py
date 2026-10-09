import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load original CSV
df = pd.read_csv("dataset.csv")

# Encode labels
le = LabelEncoder()
df["label"] = le.fit_transform(df["label"])

# Save encoded CSV
df.to_csv("dataset_encoded.csv", index=False)

# Print mapping (important)
print(dict(zip(le.classes_, le.transform(le.classes_))))

print("✅ Encoding done and saved")