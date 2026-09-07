"""

Purpose:
    First look at the raw hERG dataset BEFORE any cleaning.
    Run this once to confirm the exact SMILES column name and the
    target/label column name, then hard-code them into
    02_clean_dataset.py (and every script after it).

Output:
    Console printout only. No files are written.
"""

import pandas as pd
from pathlib import Path


# CONFIG


BASE = Path(r"D:\hERG_Prediction_Model")

# IMPORTANT: match the exact filename on disk (case-sensitive on
# some systems). Your upload was named "hERG_C_29164.csv" in one
# place and "hERG_C29164.CSV" in another -- pick ONE real filename
# and use it everywhere.
INPUT_FILE = BASE / "hERG_C_29164.csv"

 
# LOAD
df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("DATASET INSPECTION")
print("=" * 60)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 10 rows:")
print(df.head(10))

print("\nData types:")
print(df.dtypes)

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nDuplicate complete rows:")
print(df.duplicated().sum())

print("\nUnique value counts (top 20 per column):")
for col in df.columns:
    print("\n---", col, "---")
    print(df[col].value_counts(dropna=False).head(20))

print("\nDataset info:")
df.info()

print("\n" + "=" * 60)
print("NEXT STEP")
print("=" * 60)
print(
    "Look at the column list above and identify:\n"
    "  - which column holds the SMILES strings\n"
    "  - which column holds the binary hERG label (0/1)\n"
    "Then open 02_clean_dataset.py and set SMILES_COLUMN / "
    "TARGET_COLUMN to those exact names."
)

EXPECTED_COLUMNS = {"SMILES", "hERG"}

missing = EXPECTED_COLUMNS - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )
EXPECTED_COLUMNS = {"SMILES", "hERG"}

missing = EXPECTED_COLUMNS - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )

print("\nRequired columns verified:")
print("SMILES:", "SMILES" in df.columns)
print("hERG:", "hERG" in df.columns)