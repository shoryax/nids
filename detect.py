import pandas as pd
import joblib

# 1. Load trained model
rf = joblib.load("models/nids_rf_model.pkl")

# 2. Load Zeek connection logs
zeek_df = pd.read_csv("captures/conn_clean.csv")

# 3. Preprocess Zeek data
zeek_df = zeek_df.replace("-", 0)

# Load column order from training data
X_columns = pd.read_csv("datasets/X_train.csv").columns

# Ensure all columns match training features
for col in X_columns:
    if col not in zeek_df.columns:
        zeek_df[col] = 0
zeek_df = zeek_df[X_columns]

# 4. Predict
predictions = rf.predict(zeek_df)

# 5. Print results
for idx, pred in enumerate(predictions):
    status = "MALICIOUS 🚨" if pred == 1 else "Benign ✅"
    print(f"Connection {idx+1}: {status}")