"""
Purpose:
    Generate the standard evaluation figures (confusion matrix,
    ROC curve, Precision-Recall curve) for one trained model on
    the held-out test split.

Edit MODEL_FILE below to point at whichever model you want to
plot (e.g. random_forest.pkl, random_forest_tuned.pkl,
logistic_regression.pkl, svm.pkl, xgboost.pkl).

"""

import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
import matplotlib.pyplot as plt

BASE = Path(r"D:\hERG_Prediction_Model")

INPUT_FILE = BASE / "data" / "processed" / "descriptors.csv"



# EDIT THIS to the model you want to evaluate

MODEL_FILE = BASE / "models" / "random_forest_tuned.pkl"

RESULTS = BASE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

TARGET = "hERG"

df = pd.read_csv(INPUT_FILE)
X = df.drop(columns=["canonical_smiles", TARGET])
y = df[TARGET]

# Same split as every training script, so the test set matches
# what the model was actually evaluated on during training.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

model = joblib.load(MODEL_FILE)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# CONFUSION MATRIX

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title(f"Confusion Matrix — {MODEL_FILE.stem}")
plt.tight_layout()
plt.savefig(RESULTS / "confusion_matrix.png", dpi=300)
plt.show()



# ROC CURVE

RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title(f"ROC Curve — {MODEL_FILE.stem}")
plt.tight_layout()
plt.savefig(RESULTS / "roc_curve.png", dpi=300)
plt.show()



# PRECISION-RECALL CURVE

PrecisionRecallDisplay.from_predictions(y_test, y_prob)
plt.title(f"Precision-Recall Curve — {MODEL_FILE.stem}")
plt.tight_layout()
plt.savefig(RESULTS / "pr_curve.png", dpi=300)
plt.show()

print("Saved plots to:", RESULTS)