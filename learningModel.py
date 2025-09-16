import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load processed dataset
df = pd.read_csv("datasets/UNSW_NB15_processed.csv")

# 2. Separate features (X) and labels (y)
X = df.drop(columns=["label"])
y = df["label"]

# 3. First split: train vs temp (test+val)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# 4. Second split: validation vs test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# 5. Print dataset shapes
print("Train set:", X_train.shape, y_train.shape)
print("Validation set:", X_val.shape, y_val.shape)
print("Test set:", X_test.shape, y_test.shape)

# 6. Save the splits (optional)
X_train.to_csv("datasets/X_train.csv", index=False)
X_val.to_csv("datasets/X_val.csv", index=False)
X_test.to_csv("datasets/X_test.csv", index=False)
y_train.to_csv("datasets/y_train.csv", index=False)
y_val.to_csv("datasets/y_val.csv", index=False)
y_test.to_csv("datasets/y_test.csv", index=False)

from sklearn.ensemble import RandomForestClassifier
import joblib

# Create and train the model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Save the trained model
joblib.dump(rf, "models/nids_rf_model.pkl")
print("Model saved in models/nids_rf_model.pkl")