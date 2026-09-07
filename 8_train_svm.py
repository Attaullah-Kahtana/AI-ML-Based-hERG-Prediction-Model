

import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
)

BASE = Path(r"D:\hERG_Prediction_Model")

INPUT_FILE = BASE / "data" / "processed" / "descriptors.csv"
MODEL_FILE = BASE / "models" / "svm.pkl"
COMPARISON_FILE = BASE / "results" / "model_comparison.csv"

MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
COMPARISON_FILE.parent.mkdir(parents=True, exist_ok=True)

TARGET = "hERG"
MODEL_NAME = "SVM"

df = pd.read_csv(INPUT_FILE)

X = df.drop(columns=["canonical_smiles", TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

model = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "classifier",
            SVC(
                kernel="rbf",
                C=2,
                probability=True,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)

metrics = {
    "Model": MODEL_NAME,
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "Specificity": specificity,
    "F1": f1_score(y_test, y_pred, zero_division=0),
    "ROC_AUC": roc_auc_score(y_test, y_prob),
    "PR_AUC": average_precision_score(y_test, y_prob),
    "MCC": matthews_corrcoef(y_test, y_pred),
}

print(f"\n===== {MODEL_NAME} =====")
for k, v in metrics.items():
    print(f"{k}: {v}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, MODEL_FILE)
print("\nSaved model:", MODEL_FILE)

row_df = pd.DataFrame([metrics])
if COMPARISON_FILE.exists():
    existing = pd.read_csv(COMPARISON_FILE)
    existing = existing[existing["Model"] != MODEL_NAME]
    row_df = pd.concat([existing, row_df], ignore_index=True)
row_df.to_csv(COMPARISON_FILE, index=False)
print("Updated:", COMPARISON_FILE)

print(
    "\nNote: SVM training on the full descriptor set can be slow "
    "if you swap in the 2048-column fingerprint features instead "
    "-- consider LinearSVC or a smaller C search if it drags."
)