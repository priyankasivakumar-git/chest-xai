import pandas as pd

df = pd.read_csv("dataset_encoded.csv")

duplicates = df.duplicated(subset="image_path").sum()
print("Duplicates:", duplicates)