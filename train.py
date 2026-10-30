"""
JalDrushti — Drought Severity Classifier
Real Dataset: Rainfall in India 1901-2015 (IMD Meteorological Subdivisions)
Source: https://www.kaggle.com/datasets/rajanand/rainfall-in-india
Model: Random Forest Classifier
Covers: Data Cleaning & Splitting (3 Marks) + Model Training (4 Marks)
"""

import os, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json

# ─── 1. LOAD ────────────────────────────────────────────────────────────────
df = pd.read_csv("data/rainfall_india.csv")

# ─── 2. FEATURE ENGINEERING ─────────────────────────────────────────────────
# Monsoon = Jun–Sep (primary drought signal in India)
df["MONSOON"]    = df["JUN"] + df["JUL"] + df["AUG"] + df["SEP"]
df["NON_MONSOON"] = df["ANNUAL"] - df["MONSOON"]

# Subdivision normal: compute long-term mean monsoon per subdivision
sub_mean = df.groupby("SUBDIVISION")["MONSOON"].transform("mean")
df["MONSOON_DEVIATION"] = ((df["MONSOON"] - sub_mean) / sub_mean) * 100  # % deviation

# ─── 3. DERIVE DROUGHT LABEL (IMD Standard) ─────────────────────────────────
# IMD defines drought based on % deviation from normal monsoon rainfall:
#   < -25%        → Severe Drought
#   -25% to -10%  → Moderate Drought
#   -10% to +10%  → Normal
#   > +10%        → Excess
def imd_label(dev):
    if dev < -25:   return "Severe Drought"
    elif dev < -10: return "Moderate Drought"
    elif dev <= 10: return "Normal"
    else:           return "Excess Rainfall"

df["SEVERITY"] = df["MONSOON_DEVIATION"].apply(imd_label)

# ─── 4. DATA CLEANING ───────────────────────────────────────────────────────
print(f"Shape before cleaning: {df.shape}")
print(f"Nulls:\n{df[['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC','ANNUAL']].isnull().sum()}")

# Drop rows with any null rainfall values
df.dropna(subset=["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG",
                   "SEP","OCT","NOV","DEC","ANNUAL"], inplace=True)
print(f"Shape after cleaning: {df.shape}")
print(f"\nClass distribution:\n{df['SEVERITY'].value_counts()}")

# ─── 5. FEATURES & TARGET ───────────────────────────────────────────────────
FEATURES = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG",
            "SEP","OCT","NOV","DEC","MONSOON","NON_MONSOON","MONSOON_DEVIATION"]

X = df[FEATURES]
y = df["SEVERITY"]

le = LabelEncoder()
y_enc = le.fit_transform(y)

# ─── 6. SPLIT ───────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# ─── 7. TRAIN — RANDOM FOREST ───────────────────────────────────────────────
model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
model.fit(X_train_sc, y_train)

preds = model.predict(X_test_sc)
acc   = round(accuracy_score(y_test, preds) * 100, 2)
rep   = classification_report(y_test, preds, target_names=le.classes_, output_dict=True, zero_division=0)
cm    = confusion_matrix(y_test, preds).tolist()

print(f"\nAccuracy: {acc}%")
print(classification_report(y_test, preds, target_names=le.classes_, zero_division=0))

# Feature importance
importances = dict(zip(FEATURES, model.feature_importances_.round(4).tolist()))

# ─── 8. SAVE ────────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(model,  "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(le,     "models/label_encoder.pkl")

metrics = {
    "accuracy":    acc,
    "precision":   round(rep["weighted avg"]["precision"] * 100, 2),
    "recall":      round(rep["weighted avg"]["recall"] * 100, 2),
    "f1":          round(rep["weighted avg"]["f1-score"] * 100, 2),
    "cm":          cm,
    "cm_labels":   le.classes_.tolist(),
    "features":    FEATURES,
    "importances": importances,
    "model_name":  "Random Forest (n=100, depth=8)",
    "dataset_info": {
        "source": "Rainfall in India 1901-2015, IMD Meteorological Subdivisions",
        "rows":   int(df.shape[0]),
        "features": len(FEATURES),
        "classes": le.classes_.tolist()
    }
}
with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nAll artifacts saved to models/")
