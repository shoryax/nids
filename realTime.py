#!/usr/bin/env python3
"""
realtime_detect.py
Monitors Zeek's conn.log in real time, preprocesses new lines into feature vectors,
uses trained model to predict malicious/benign, and raises alerts.
"""

import time
import joblib
import pandas as pd
import subprocess
import os
from datetime import datetime

# CONFIG
ZEEK_LOG = "captures/conn.log"
MODEL_PATH = "models/nids_rf_model.pkl"
FEATURES_PATH = "models/features.pkl"
ENCODERS_PATH = "models/encoders.pkl"
ALERT_LOG = "alerts.log"
DO_DESKTOP_NOTIFY = True


def desktop_notify(title, message):
    """Show a macOS desktop notification (no-op on errors)."""
    if not DO_DESKTOP_NOTIFY:
        return
    safe_message = message.replace('"', '\\"')
    cmd = ["osascript", "-e", f'display notification "{safe_message}" with title "{title}"']
    try:
        subprocess.run(cmd, check=False)
    except Exception as e:
        print("Notification failed:", e)


def load_artifacts():
    """Load model, feature names, and optional encoders."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Error: Model file not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Error: Feature names file not found at {FEATURES_PATH}")
    feature_names = joblib.load(FEATURES_PATH)

    encoders = None
    if os.path.exists(ENCODERS_PATH):
        encoders = joblib.load(ENCODERS_PATH)
    else:
        print("Warning: Encoders file not found. Falling back to hash mapping for categoricals.")

    return model, feature_names, encoders


def parse_zeek_line(line, header_fields):
    """Parse a Zeek conn.log line into {field: value} dict."""
    parts = line.strip().split("\t")
    if len(parts) != len(header_fields):
        return None
    return dict(zip(header_fields, parts))


def safe_num(x):
    """Convert Zeek numeric field (or '-') into float/int safely."""
    try:
        if x in ("-", None, ""):
            return 0
        if "." in str(x):
            return float(x)
        return int(x)
    except Exception:
        return 0


def make_feature_row(parsed, feature_names, encoders):
    """Turn parsed Zeek fields into DataFrame row matching training features."""
    row = {}
    for f in feature_names:
        val = parsed.get(f, 0)

        if val in ("-", None, ""):
            row[f] = 0
        elif f.lower() in (
            "duration", "orig_bytes", "resp_bytes", "total_bytes",
            "sbytes", "dbytes", "spkts", "dpkts", "sload", "dload",
            "orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes"
        ):
            row[f] = safe_num(val)
        else:
            # categorical fields
            if encoders and f in encoders:
                enc = encoders[f]
                try:
                    row[f] = enc.transform([val])[0]
                except Exception:
                    row[f] = -1  # unseen category
            else:
                # fallback: map to int via hash for consistency
                row[f] = abs(hash(val)) % 1000

    df = pd.DataFrame([row], columns=feature_names)
    return df


def watch_conn_log(model, feature_names, encoders):
    """Tail Zeek conn.log and run predictions on new entries."""
    processed_uids = set()
    print("Waiting for Zeek conn.log...")

    while not os.path.exists(ZEEK_LOG):
        time.sleep(1)

    header_fields = None
    with open(ZEEK_LOG, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("#fields"):
                header_fields = line.strip().split()[1:]
                break
    if not header_fields:
        print("Warning: '#fields' not found. Using fallback.")
        header_fields = [
            "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
            "proto", "service", "duration", "orig_bytes", "resp_bytes", "conn_state"
        ]
    print("Zeek conn.log fields:", header_fields)

    with open(ZEEK_LOG, "r", encoding="utf-8", errors="ignore") as fh:
        fh.seek(0, os.SEEK_END)

        while True:
            line = fh.readline()
            if not line:
                time.sleep(0.5)
                continue
            if line.startswith("#") or not line.strip():
                continue

            parsed = parse_zeek_line(line, header_fields)
            if not parsed:
                continue

            uid = parsed.get("uid")
            if uid and uid in processed_uids:
                continue
            if uid:
                processed_uids.add(uid)

            row_df = make_feature_row(parsed, feature_names, encoders)

            try:
                pred = model.predict(row_df)[0]
                score = None
                if hasattr(model, "predict_proba"):
                    score = model.predict_proba(row_df)[0].max()

                src = parsed.get("id.orig_h", "-")
                dst = parsed.get("id.resp_h", "-")
                proto = parsed.get("proto", "-")
                service = parsed.get("service", "-")

                if pred == 1:
                    msg = (f"[{datetime.now()}] ALERT: Malicious "
                           f"UID={uid} {src}->{dst} proto={proto} service={service} score={score}")
                    print(msg)
                    with open(ALERT_LOG, "a") as af:
                        af.write(msg + "\n")
                    desktop_notify("NIDS Alert", f"{src} -> {dst} ({service})")
                else:
                    print(f"[{datetime.now()}] OK: UID={uid} {src}->{dst} {service}")

            except Exception as e:
                print("Prediction failed:", e)


def main():
    model, feature_names, encoders = load_artifacts()
    if hasattr(feature_names, "tolist"):
        feature_names = feature_names.tolist()
    print("Monitoring Zeek conn.log...")
    watch_conn_log(model, feature_names, encoders)


if __name__ == "__main__":
    main()