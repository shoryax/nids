import pandas as pd

# Load CSVs
zeek_df = pd.read_csv("captures/conn.csv", sep="\t")
unsw_df = pd.read_csv("datasets/UNSW_NB15_training-set.csv")

# Drop unnecessary columns
zeek_df = zeek_df.drop(columns=[
    "ts", "uid", "id.orig_h", "id.resp_h", "id.orig_p", "id.resp_p"
], errors='ignore')

unsw_df = unsw_df.drop(columns=[
    "srcip", "sport", "dstip", "dsport"
], errors='ignore')

# Replace '-' with 0
zeek_df.replace("-", 0, inplace=True)
unsw_df.replace("-", 0, inplace=True)

# Save cleaned data
zeek_df.to_csv("captures/conn_clean.csv", index=False)
unsw_df.to_csv("datasets/UNSW_NB15_clean.csv", index=False)

# Print columns
print("Zeek Columns:", zeek_df.columns.tolist())
print("UNSW Columns:", unsw_df.columns.tolist())