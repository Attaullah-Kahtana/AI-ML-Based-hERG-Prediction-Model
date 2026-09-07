"""

Purpose:
    Compute a set of classic physicochemical / Lipinski-style
    RDKit descriptors for every molecule. These become the
    feature matrix for the Logistic Regression / RandomForest /
    SVM / XGBoost models.

"""

import pandas as pd
import numpy as np
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors

BASE = Path(r"D:\hERG_Prediction_Model")

INPUT_FILE = BASE / "data" / "processed" / "cleaned_dataset.csv"
OUTPUT_FILE = BASE / "data" / "processed" / "descriptors.csv"

SMILES_COLUMN = "canonical_smiles"
TARGET_COLUMN = "hERG"

df = pd.read_csv(INPUT_FILE)


# DESCRIPTOR FUNCTIONS


descriptor_functions = {
    "MolWt": Descriptors.MolWt,
    "LogP": Crippen.MolLogP,
    "TPSA": rdMolDescriptors.CalcTPSA,
    "HBD": Lipinski.NumHDonors,
    "HBA": Lipinski.NumHAcceptors,
    "RotatableBonds": Lipinski.NumRotatableBonds,
    "RingCount": Lipinski.RingCount,
    "HeavyAtomCount": Lipinski.HeavyAtomCount,
    "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
    "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings,
    "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings,
    "NumHeteroatoms": Lipinski.NumHeteroatoms,
    "NumSaturatedRings": rdMolDescriptors.CalcNumSaturatedRings,
    "NumAromaticHeterocycles": rdMolDescriptors.CalcNumAromaticHeterocycles,
    "NumAliphaticHeterocycles": rdMolDescriptors.CalcNumAliphaticHeterocycles,
}


# CALCULATE

all_descriptors = []

for smiles in df[SMILES_COLUMN]:
    mol = Chem.MolFromSmiles(smiles)
    row = {}
    for name, function in descriptor_functions.items():
        try:
            row[name] = function(mol)
        except Exception:
            row[name] = np.nan
    all_descriptors.append(row)

descriptor_df = pd.DataFrame(all_descriptors)



# COMBINE + SAVE

final_df = pd.concat(
    [df[[SMILES_COLUMN, TARGET_COLUMN]].reset_index(drop=True), descriptor_df],
    axis=1,
)

final_df.to_csv(OUTPUT_FILE, index=False)

print("Descriptor dataset:", final_df.shape)
print("\nFirst rows:")
print(final_df.head())
print("\nMissing descriptor values:")
print(final_df.isnull().sum())
print("\nSaved:", OUTPUT_FILE)