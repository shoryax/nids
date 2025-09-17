import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1. Load the cleaned UNSW dataset
unsw_df = pd.read_csv("datasets/UNSW_NB15_clean.csv")

# 2. Identify the label column (it's often called 'label' or 'attack_cat')
print(unsw_df.columns)

# If 'attack_cat' exists, map it to binary 'label'
if 'attack_cat' in unsw_df.columns:
    unsw_df['label'] = unsw_df['attack_cat'].apply(lambda x: 0 if x == 'Normal' else 1)
elif 'label' in unsw_df.columns:
    # If it's already binary, ensure it's int
    unsw_df['label'] = unsw_df['label'].astype(int)

# 3. Drop rows with missing values (if any remain)
unsw_df.dropna(inplace=True)

# 4. Encode categorical columns (like proto, service, state)
categorical_cols = unsw_df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    unsw_df[col] = le.fit_transform(unsw_df[col])

# 5. Save this fully processed dataset
unsw_df.to_csv("datasets/UNSW_NB15_processed.csv", index=False)

print("Processed UNSW dataset saved with numeric features & binary labels!")
print("Shape:", unsw_df.shape)