

import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

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
MODEL_FILE = BASE / "models" / "random_forest_tuned.pkl"
COMPARISON_FILE = BASE / "results" / "model_comparison.csv"

MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
COMPARISON_FILE.parent.mkdir(parents=True, exist_ok=True)

TARGET = "hERG"
MODEL_NAME = "RandomForest_Tuned"

df = pd.read_csv(INPUT_FILE)

X = df.drop(columns=["canonical_smiles", TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

base_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        (
            "classifier",
            RandomForestClassifier(
                class_weight="balanced", random_state=42, n_jobs=-1
            ),
        ),
    ]
)

# Parameter names need the "classifier__" prefix since the
# estimator being searched is a Pipeline, not a bare RandomForest.
param_grid = {
    "classifier__n_estimators": [300, 500, 800],
    "classifier__max_depth": [None, 10, 20, 30],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__max_features": ["sqrt", "log2"],
}

search = RandomizedSearchCV(
    estimator=base_pipeline,
    param_distributions=param_grid,
    n_iter=20,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
    random_state=42,
    verbose=1,
)

search.fit(X_train, y_train)

print("Best parameters:", search.best_params_)
print("Best CV ROC-AUC:", search.best_score_)

best_model = search.best_estimator_

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

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

print(f"\n===== {MODEL_NAME} (held-out test set) =====")
for k, v in metrics.items():
    print(f"{k}: {v}")

joblib.dump(best_model, MODEL_FILE)
print("\nSaved model:", MODEL_FILE)

row_df = pd.DataFrame([metrics])
if COMPARISON_FILE.exists():
    existing = pd.read_csv(COMPARISON_FILE)
    existing = existing[existing["Model"] != MODEL_NAME]
    row_df = pd.concat([existing, row_df], ignore_index=True)
row_df.to_csv(COMPARISON_FILE, index=False)
print("Updated:", COMPARISON_FILE)

from sklearn.model_selection import StratifiedKFold

cv_strategy = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

search = RandomizedSearchCV(
    estimator=base_pipeline,
    param_distributions=param_grid,
    n_iter=20,
    scoring="roc_auc",
    cv=cv_strategy,
    n_jobs=-1,
    random_state=42,
    verbose=1,
)