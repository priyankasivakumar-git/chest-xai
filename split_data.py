import pandas as pd
from sklearn.model_selection import train_test_split

# Load encoded dataset
df = pd.read_csv("dataset.csv")

# Split (70% train, 30% temp)
train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    stratify=df["label"],
    random_state=42
)

# Split temp → val (15%) + test (15%)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label"],
    random_state=42
)

print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))