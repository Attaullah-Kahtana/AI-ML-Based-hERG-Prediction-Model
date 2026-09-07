"""
Purpose:
    Compute Morgan (ECFP-like) fingerprints as an alternative
    feature representation to the physicochemical descriptors.
    Not required for the current training scripts (which use
    descriptors.csv) but kept as an optional feature set you can
    swap in later for comparison.

Input:
    <BASE>/data/processed/cleaned_dataset.csv

Output:
    <BASE>/data/processed/fingerprints.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs

BASE = Path(r"D:\hERG_Prediction_Model")

INPUT_FILE = BASE / "data" / "processed" / "cleaned_dataset.csv"
OUTPUT_FILE = BASE / "data" / "processed" / "fingerprints.csv"

SMILES_COLUMN = "canonical_smiles"
TARGET_COLUMN = "hERG"

df = pd.read_csv(INPUT_FILE)

generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

fingerprints = []

for smiles in df[SMILES_COLUMN]:
    mol = Chem.MolFromSmiles(smiles)
    fp = generator.GetFingerprint(mol)
    arr = np.zeros(2048, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    fingerprints.append(arr)

X = np.array(fingerprints)
columns = [f"FP_{i}" for i in range(2048)]

fp_df = pd.DataFrame(X, columns=columns)
fp_df.insert(0, SMILES_COLUMN, df[SMILES_COLUMN].values)
fp_df[TARGET_COLUMN] = df[TARGET_COLUMN].values

fp_df.to_csv(OUTPUT_FILE, index=False)

print("Fingerprint matrix:", X.shape)
print("\nSaved:", OUTPUT_FILE)