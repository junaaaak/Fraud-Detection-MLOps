import pandas as pd
import joblib

from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score)


df = pd.read_csv("data/transactions.csv")

FEATURES = [
    "amount",
    "txn_count_5m",
    "amount_sum_1h",
    "amount_ratio",
    "city_changed"]

X = df[FEATURES]
y = df["fraud_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42))])

model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("ROC AUC Score:", round(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]), 4))

#save the model
Path("models").mkdir(parents=True, exist_ok=True)

artifact = {
    "model": model,
    "features": FEATURES,
    "version": "v1.0"
}

joblib.dump(artifact, "models/fraud_detection_model.pkl")

print("Model training completed.")
print ("Model saved to 'models/fraud_detection_model.pkl'")