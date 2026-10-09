import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Load dataset
df = pd.read_csv("dataset.csv")

# Recreate split (same as before)
from sklearn.model_selection import train_test_split

train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    stratify=df["label"],
    random_state=42
)

# 🔥 STEP 5: CLASS WEIGHTS
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df["label"]),
    y=train_df["label"]
)

class_weights = dict(enumerate(class_weights))
class_weights = {k: float(v) for k, v in class_weights.items()}
print("Class Weights:", class_weights)