
"""
FAST SHAP EXPLANATION FOR hERG RANDOM FOREST

Purpose:
    Explain which molecular descriptors push predictions toward
    hERG blocker (1) or non-blocker (0).

Speed optimization:
    - Uses only a representative sample of the test set.
    - Avoids unnecessary DataFrame conversions.
    - Uses TreeExplainer directly on the trained tree model.
    - Does not display the plot interactively.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

BASE = Path(r"D:\hERG_Prediction_Model")

INPUT_FILE = BASE / "data" / "processed" / "descriptors.csv"

MODEL_FILE = BASE / "models" / "random_forest_tuned.pkl"

RESULTS = BASE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

TARGET = "hERG"

OUTPUT_FILE = RESULTS / "shap_summary.png"


# ============================================================
# SPEED SETTINGS
# ============================================================

# Number of compounds used for SHAP.
# 300–500 is usually enough for a fast summary plot.
SHAP_SAMPLES = 500

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FAST SHAP ANALYSIS")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Dataset: {len(df):,} compounds")


# ============================================================
# PREPARE X AND y
# ============================================================

X = df.drop(columns=["canonical_smiles", TARGET])
y = df[TARGET]


# ============================================================
# SAME TRAIN/TEST SPLIT USED DURING MODEL TRAINING
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

print(f"Test set: {len(X_test):,} compounds")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

pipeline = joblib.load(MODEL_FILE)

imputer = pipeline.named_steps["imputer"]
classifier = pipeline.named_steps["classifier"]


# ============================================================
# SAMPLE TEST DATA FOR SHAP
# ============================================================

sample_size = min(SHAP_SAMPLES, len(X_test))

X_shap = X_test.sample(
    n=sample_size,
    random_state=RANDOM_STATE
)

print(f"SHAP sample: {len(X_shap):,} compounds")


# ============================================================
# IMPUTE ONLY THE SHAP SAMPLE
# ============================================================

print("\nPreparing data...")

X_shap_imputed = imputer.transform(X_shap)

# Convert back to DataFrame so SHAP keeps descriptor names
X_shap_imputed = pd.DataFrame(
    X_shap_imputed,
    columns=X_shap.columns,
    index=X_shap.index
)


# ============================================================
# SHAP TREE EXPLAINER
# ============================================================

print("\nCalculating SHAP values...")

explainer = shap.TreeExplainer(
    classifier,
    feature_perturbation="tree_path_dependent"
)

shap_values = explainer.shap_values(X_shap_imputed)


# ============================================================
# HANDLE BINARY CLASSIFICATION
# ============================================================

if isinstance(shap_values, list):

    # Class 1 = hERG blocker
    shap_values_to_plot = shap_values[1]

else:

    # Newer SHAP versions may return a NumPy array
    shap_values_to_plot = shap_values

    # Handle shape:
    # (samples, features, classes)
    if len(shap_values_to_plot.shape) == 3:
        shap_values_to_plot = shap_values_to_plot[:, :, 1]


# ============================================================
# CREATE SUMMARY PLOT
# ============================================================

print("\nCreating SHAP plot...")

plt.figure(figsize=(10, 7))

shap.summary_plot(
    shap_values_to_plot,
    X_shap_imputed,
    show=False,
    max_display=15
)

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("SHAP ANALYSIS COMPLETE")
print("=" * 70)

print(f"SHAP compounds used : {sample_size:,}")
print(f"Descriptors analyzed: {X.shape[1]:,}")
print(f"Output              : {OUTPUT_FILE}")
print("=" * 70)
