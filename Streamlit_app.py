import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors


st.set_page_config(
    page_title="hERG Prediction Model",
    page_icon="🧪",
    layout="wide",
)


@st.cache_resource
def load_model():
    """Load the first available trained model."""
    model_path = os.getenv("MODEL_PATH")

    candidates = [
        model_path,
        "model.pkl",
        "model.joblib",
        "hERG_model.pkl",
        "hERG_model.joblib",
        "models/model.pkl",
        "models/model.joblib",
    ]

    for path in candidates:
        if path and Path(path).exists():
            return joblib.load(path), path

    return None, None


def calculate_descriptors(smiles):
    """Convert a SMILES string into RDKit molecular descriptors."""
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        raise ValueError("Invalid SMILES string.")

    descriptor_values = []
    descriptor_names = []

    for name, function in Descriptors.descList:
        try:
            value = function(molecule)
            descriptor_values.append(float(value))
            descriptor_names.append(name)
        except Exception:
            descriptor_values.append(0.0)
            descriptor_names.append(name)

    values = np.nan_to_num(
        np.asarray(descriptor_values, dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return values, descriptor_names


def prepare_features(model, smiles):
    """Prepare descriptor features for the loaded model."""
    descriptors, names = calculate_descriptors(smiles)

    expected_features = getattr(model, "n_features_in_", None)

    if expected_features is not None:
        expected_features = int(expected_features)

        if len(descriptors) < expected_features:
            descriptors = np.pad(
                descriptors,
                (0, expected_features - len(descriptors)),
                constant_values=0,
            )
        else:
            descriptors = descriptors[:expected_features]

    return pd.DataFrame([descriptors], columns=names[: len(descriptors)])


def predict(model, smiles):
    """Generate a prediction and optional probability."""
    features = prepare_features(model, smiles)

    try:
        prediction = model.predict(features)[0]
    except Exception:
        # Supports models or pipelines that directly accept SMILES strings.
        prediction = model.predict([smiles])[0]

    probability = None

    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(features)[0]
            probability = float(np.max(probabilities))
        except Exception:
            probability = None

    return prediction, probability


st.title("🧪 hERG Cardiotoxicity Prediction")
st.write(
    "Enter a molecule as a SMILES string to predict its hERG activity."
)

model, model_path = load_model()

if model is None:
    st.error(
        "No trained model was found. Add a model file named "
        "`model.pkl` or `model.joblib` to the repository."
    )
    st.stop()

st.success(f"Loaded model: `{model_path}`")

with st.sidebar:
    st.header("Input")
    smiles = st.text_input(
        "SMILES string",
        value="CCO",
        help="Example: CCO represents ethanol.",
    )

    predict_button = st.button(
        "Predict",
        type="primary",
        use_container_width=True,
    )

if predict_button:
    try:
        prediction, probability = predict(model, smiles)

        st.subheader("Prediction")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Predicted class", str(prediction))

        with col2:
            if probability is not None:
                st.metric("Confidence", f"{probability:.2%}")

        st.success("Prediction completed successfully.")

        descriptor_values, descriptor_names = calculate_descriptors(smiles)

        with st.expander("Molecular descriptors"):
            descriptor_table = pd.DataFrame(
                {
                    "Descriptor": descriptor_names,
                    "Value": descriptor_values,
                }
            )
            st.dataframe(
                descriptor_table,
                use_container_width=True,
                hide_index=True,
            )

    except Exception as error:
        st.error(f"Prediction failed: {error}")


st.divider()
st.caption(
    "This application is intended for research and educational use only."
)
