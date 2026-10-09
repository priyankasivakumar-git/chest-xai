import os
import pandas as pd
import random

dataset_path = r"C:\Users\raghul.s\OneDrive\Desktop\Priyanka_Akka_Project\dataset"

data = []

for label in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, label)

    for img in os.listdir(class_path):
        if img.lower().endswith((".png", ".jpg", ".jpeg")):
            data.append({
                "image_path": os.path.join(class_path, img),
                "label": label,
                "age": random.randint(20, 80),
                "gender": random.choice([0, 1]),
                "fever": random.choice([0, 1]),
                "cough": random.choice([0, 1])
            })

df = pd.DataFrame(data)
df.to_csv("dataset.csv", index=False)

print("✅ Correct CSV created")