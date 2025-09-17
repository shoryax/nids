import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load test data
X_test = pd.read_csv("datasets/X_test.csv")
y_test = pd.read_csv("datasets/y_test.csv").values.ravel()

# Load trained model
rf = joblib.load("models/nids_rf_model.pkl")

# 2. Predict on test data
y_test_pred = rf.predict(X_test)

# 3. Print evaluation metrics
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("Classification Report:\n", classification_report(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

# 4. Feature Importance
importances = rf.feature_importances_
feature_names = X_test.columns

indices = importances.argsort()[::-1]

# 5. Plot feature importance
plt.figure(figsize=(12, 6))
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), feature_names[indices], rotation=90)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()