"""
02_clean_dataset.py

Purpose:
    Take the raw CSV, keep only the SMILES + target columns,
    drop missing values, validate every SMILES with RDKit,
    canonicalize them, and remove duplicate structures.

"""

import pandas as pd
from pathlib import Path
from rdkit import Chem

BASE = Path(r"D:\hERG_Prediction_Model")

INPUT_FILE = BASE / "hERG_C_29164.csv"
OUTPUT_FILE = BASE / "data" / "processed" / "cleaned_dataset.csv"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)



# SET THESE FROM WHAT 01_data_inspection.py PRINTED

SMILES_COLUMN = "SMILES"
TARGET_COLUMN = "hERG"


# LOAD

df = pd.read_csv(INPUT_FILE)
print("Original dataset:", df.shape)

df = df[[SMILES_COLUMN, TARGET_COLUMN]].copy()


# DROP MISSING VALUES

print("\nMissing values before drop:")
print(df.isnull().sum())

df = df.dropna(subset=[SMILES_COLUMN, TARGET_COLUMN])



# CLEAN SMILES TEXT

df[SMILES_COLUMN] = df[SMILES_COLUMN].astype(str).str.strip()



# CONVERT TARGET TO NUMERIC / BINARY

df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
df = df.dropna(subset=[TARGET_COLUMN])

# Only allow binary labels
df = df[df[TARGET_COLUMN].isin([0, 1])]
df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)



# VALIDATE SMILES WITH RDKit


def parse_smiles(smiles):
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


df["Molecule"] = df[SMILES_COLUMN].apply(parse_smiles)

invalid = df["Molecule"].isna().sum()
print("\nInvalid SMILES removed:", invalid)

df = df[df["Molecule"].notna()].copy()


# CANONICAL SMILES

df["canonical_smiles"] = df["Molecule"].apply(
    lambda mol: Chem.MolToSmiles(mol, canonical=True)
)


# REMOVE DUPLICATE STRUCTURES

# ============================================================
# CHECK DUPLICATE STRUCTURES WITH CONFLICTING LABELS
# ============================================================

duplicate_groups = (
    df.groupby("canonical_smiles")[TARGET_COLUMN]
    .nunique()
)

conflicting_structures = duplicate_groups[
    duplicate_groups > 1
]

print(
    "\nStructures with conflicting hERG labels:",
    len(conflicting_structures)
)

if len(conflicting_structures) > 0:
    print("\nWARNING:")
    print(
        "Some identical canonical SMILES have different hERG labels."
    )

    conflict_smiles = conflicting_structures.index.tolist()

    print("\nExample conflicting structures:")
    print(
        df[
            df["canonical_smiles"].isin(conflict_smiles)
        ][
            [SMILES_COLUMN, "canonical_smiles", TARGET_COLUMN]
        ].head(20)
    )

    # Remove ambiguous structures rather than arbitrarily
    # choosing one label.
    df = df[
        ~df["canonical_smiles"].isin(conflict_smiles)
    ].copy()

# ============================================================
# REMOVE NON-CONFLICTING DUPLICATE STRUCTURES
# ============================================================

before = len(df)

df = df.drop_duplicates(
    subset=["canonical_smiles"],
    keep="first"
)

print(
    "Duplicate structures removed:",
    before - len(df)
)


# SAVE (drop the RDKit Mol object first, it isn't CSV-safe)

df = df[[SMILES_COLUMN, "canonical_smiles", TARGET_COLUMN]]

df.to_csv(OUTPUT_FILE, index=False)

print("\nFinal dataset:", df.shape)

print("\nClass distribution:")
print(df[TARGET_COLUMN].value_counts())

print("\nClass proportions:")
print(df[TARGET_COLUMN].value_counts(normalize=True))

print("\nSaved:", OUTPUT_FILE)