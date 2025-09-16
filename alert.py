import pandas as pd
import joblib
from datetime import datetime

# Load saved model and feature order
rf = joblib.load("models/nids_model.pkl")
features = joblib.load("models/features.pkl")

# Load Zeek connections
conn_df = pd.read_csv("captures/conn_clean.csv")

# Predict
preds = rf.predict(conn_df[features])

# Open alert log
with open("alerts.log", "a") as f:
    for i, pred in enumerate(preds):
        if pred == 1:  # Assuming 1 = malicious
            alert_msg = f"[{datetime.now()}] ALERT: Suspicious connection at row {i}\n"
            print(alert_msg.strip())
            f.write(alert_msg)