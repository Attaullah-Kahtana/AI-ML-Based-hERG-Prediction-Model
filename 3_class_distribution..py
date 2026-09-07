"""

Purpose:
    Analyze the balance of hERG blocker and non-blocker classes.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE = Path(r"D:\hERG_Prediction_Model")

INPUT_FILE = (
    BASE
    / "data"
    / "processed"
    / "cleaned_dataset.csv"
)

RESULTS = BASE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

TARGET = "hERG"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found."
    )


# ============================================================
# CLASS COUNTS
# ============================================================

counts = df[TARGET].value_counts().sort_index()

percentages = (
    df[TARGET]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
)


print("=" * 60)
print("hERG CLASS DISTRIBUTION")
print("=" * 60)

print("\nClass counts:")
print(counts)

print("\nClass percentages:")
print(percentages.round(2))


# ============================================================
# PLOT
# ============================================================

labels = [
    "Non-blocker (0)",
    "Blocker (1)"
]

values = [
    counts.get(0, 0),
    counts.get(1, 0)
]

plt.figure(figsize=(7, 5))

plt.bar(labels, values)

plt.title("hERG Activity Class Distribution")
plt.xlabel("hERG Class")
plt.ylabel("Number of Compounds")

plt.tight_layout()

OUTPUT_FILE = RESULTS / "class_distribution.png"

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSaved:", OUTPUT_FILE)