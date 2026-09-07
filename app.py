import os
import gradio as gr
import pandas as pd
import numpy as np
import joblib

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors

from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path(__file__).resolve().parent

MODEL_FILE = BASE / "models" / "random_forest_tuned.pkl"


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = joblib.load(MODEL_FILE)


# ============================================================
# SAME DESCRIPTORS USED DURING TRAINING
# ============================================================

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


# ============================================================
# DESCRIPTOR CALCULATION
# ============================================================

def calculate_descriptors(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None, None

    values = {}

    for name, function in descriptor_functions.items():

        try:
            values[name] = function(mol)

        except Exception:
            values[name] = np.nan

    descriptors = pd.DataFrame([values])

    return mol, descriptors


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_herg(smiles):

    if smiles is None or not smiles.strip():

        return (
            "Please enter a SMILES string.",
            "",
            None,
            ""
        )

    smiles = smiles.strip()

    mol, X = calculate_descriptors(smiles)

    if mol is None:

        return (
            "Invalid SMILES",
            "",
            None,
            "RDKit could not parse the supplied SMILES string."
        )

    try:

        prediction = int(model.predict(X)[0])

        probability = float(
            model.predict_proba(X)[0][1]
        )

    except Exception as e:

        return (
            "Prediction error",
            "",
            None,
            str(e)
        )

    probability_percent = probability * 100

    if prediction == 1:

        label = "⚠️ hERG BLOCKER / HIGHER RISK"

    else:

        label = "✅ NON-BLOCKER / LOWER PREDICTED RISK"

    descriptor_table = X.T.reset_index()

    descriptor_table.columns = [
        "Descriptor",
        "Value"
    ]

    explanation = (
        f"Predicted class: {prediction}\n"
        f"hERG blocker probability: "
        f"{probability_percent:.2f}%\n\n"
        "This is an in-silico machine-learning prediction "
        "and should not be interpreted as a clinical diagnosis "
        "or experimental confirmation."
    )

    return (
        label,
        f"{probability_percent:.2f}%",
        descriptor_table,
        explanation
    )


# ============================================================
# GRADIO INTERFACE
# ============================================================

description = """
## AI-Based hERG Effect Prediction

Enter a molecular SMILES string to predict the probability
of hERG-related cardiotoxicity using a trained machine-learning
model.

### Model
Tuned Random Forest trained using RDKit molecular descriptors.

### Output
- Predicted hERG class
- hERG blocker probability
- Molecular descriptors

**Important:** This tool is for research/educational purposes only.
Predictions should be experimentally validated and must not be used
as a clinical diagnosis.
"""


with gr.Blocks(title="AI-Based hERG Effect Prediction") as demo:

    gr.Markdown(
        "# 🧬 AI-Based hERG Effect Prediction"
    )

    gr.Markdown(description)

    smiles_input = gr.Textbox(
        label="SMILES",
        placeholder="Example: CC(=O)OC1=CC=CC=C1C(=O)O",
        lines=2
    )

    predict_button = gr.Button(
        "Predict hERG Effect"
    )

    prediction_output = gr.Textbox(
        label="Prediction"
    )

    probability_output = gr.Textbox(
        label="hERG Blocker Probability"
    )

    descriptors_output = gr.Dataframe(
        label="Calculated Molecular Descriptors"
    )

    explanation_output = gr.Textbox(
        label="Details",
        lines=6
    )

    predict_button.click(
        fn=predict_herg,
        inputs=smiles_input,
        outputs=[
            prediction_output,
            probability_output,
            descriptors_output,
            explanation_output
        ]
    )

    gr.Examples(
        examples=[
            ["CC(=O)OC1=CC=CC=C1C(=O)O"],
            ["CN1CCC[C@H]1c1cccnc1"],
        ],
        inputs=smiles_input
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        ssr_mode=False
    )