import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("datasets/UNSW_NB15_processed.csv")

"""
Prepare features/labels
- Drop label column from features
- Also drop identifier and known leakage columns if present
"""
# Separate features and labels
X = df.drop(columns=["label", "attack_cat", "id"], errors="ignore")
y = df["label"]

from sklearn.preprocessing import LabelEncoder

# Only encode categorical/object dtype columns if they exist
candidate_cats = ["proto", "service", "state"]  # use 'state' (not 'conn_state')
encoders = {}
for col in candidate_cats:
    if col in X.columns and X[col].dtype == object:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

if encoders:
    joblib.dump(encoders, "models/encoders.pkl")

# Split into train and temp (val+test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Split temp into validation and test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# Save splits
X_train.to_csv("datasets/X_train.csv", index=False)
X_val.to_csv("datasets/X_val.csv", index=False)
X_test.to_csv("datasets/X_test.csv", index=False)
y_train.to_csv("datasets/y_train.csv", index=False)
y_val.to_csv("datasets/y_val.csv", index=False)
y_test.to_csv("datasets/y_test.csv", index=False)

# Train Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Save model, feature names, and encoders
joblib.dump(rf, "models/nids_rf_model.pkl")
joblib.dump(X_train.columns.tolist(), "models/features.pkl")
if encoders:
    joblib.dump(encoders, "models/encoders.pkl")

print("Training complete and artifacts saved.")
