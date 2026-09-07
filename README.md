
# AI-ML-Based-hERG-Prediction-Model


### Machine Learning & AI-Based Prediction of hERG-Related Cardiotoxicity from Chemical Structures

AI & ML-based project for predicting **hERG-related cardiotoxicity** from molecular structures represented as SMILES.

The project uses **RDKit molecular descriptors** and multiple machine-learning algorithms to classify compounds as potential **hERG blockers** or **non-blockers**. A tuned Random Forest model is deployed through an interactive **Gradio** web application.

> **Research/educational tool:** This application is intended for computational research, screening, and educational purposes. Predictions should not be interpreted as clinical diagnoses or definitive experimental evidence.

---

## 🚀 Live Demo

**Hugging Face Space:**
Add your deployed Hugging Face Space URL here.

**GitHub Repository:**
https

**Developer:**
Attaullah

**LinkedIn:**
https://www.linkedin.com/in/atta-ullah-67a310331/

---

## 📌 Project Overview

The human ether-à-go-go-related gene (**hERG**) potassium channel is an important target in drug-safety assessment because compounds that interfere with hERG channel activity can potentially contribute to cardiac arrhythmias.

Early computational prediction of hERG activity can therefore help prioritize compounds for further experimental investigation.

This project develops a machine-learning pipeline that:

1. Loads and inspects a molecular dataset.
2. Cleans and validates SMILES structures.
3. Generates molecular descriptors using RDKit.
4. Trains multiple machine-learning models.
5. Compares model performance.
6. Tunes the Random Forest model.
7. Evaluates the final model.
8. Performs SHAP-based model interpretation.
9. Deploys the trained model as an interactive web application.

---

## 🧬 Dataset

The project uses a dataset containing approximately **29,164 compounds** with:

* **SMILES** — molecular structure representation
* **hERG** — binary activity label

### Target Definition

| Label | Meaning          |
| ----: | ---------------- |
|   `0` | hERG non-blocker |
|   `1` | hERG blocker     |

The raw dataset is processed before model training to remove invalid and unsuitable molecular records.

---

# 🧪 Molecular Descriptors

The current model uses **15 RDKit molecular descriptors**:

1. Molecular Weight
2. LogP
3. Topological Polar Surface Area (TPSA)
4. Hydrogen Bond Donors
5. Hydrogen Bond Acceptors
6. Rotatable Bonds
7. Ring Count
8. Heavy Atom Count
9. Fraction CSP3
10. Number of Aromatic Rings
11. Number of Aliphatic Rings
12. Number of Heteroatoms
13. Number of Saturated Rings
14. Number of Aromatic Heterocycles
15. Number of Aliphatic Heterocycles

These descriptors convert molecular structures into numerical features that can be processed by machine-learning algorithms.

---

# 🤖 Machine Learning Models

Four baseline models are implemented:

### 1. Logistic Regression

A linear classification model used as a baseline.

### 2. Random Forest

An ensemble of decision trees capable of modeling nonlinear relationships between molecular descriptors and hERG activity.

### 3. Support Vector Machine

An RBF-kernel SVM is used to model nonlinear decision boundaries.

### 4. XGBoost

A gradient-boosting algorithm used as another powerful nonlinear classification approach.

---


# 📊 Model Evaluation

The models are evaluated using multiple classification metrics:

* Accuracy
* Precision
* Recall / Sensitivity
* Specificity
* F1-score
* ROC-AUC
* PR-AUC
* Matthews Correlation Coefficient (MCC)

Additional evaluation includes:

* Confusion matrix
* ROC curve
* Precision-Recall curve
* Feature importance
* SHAP analysis

Using multiple metrics is important because biological classification datasets can have class imbalance, making accuracy alone insufficient.

---


# 🖥️ Application Features

* 🧬 SMILES input
* 🔬 RDKit molecular validation
* 📊 Automatic descriptor calculation
* 🤖 Machine-learning prediction
* 🌲 Tuned Random Forest model
* 📋 Molecular descriptor table
* ⚡ Interactive Gradio interface
* ☁️ Hugging Face deployment
* 📱 Browser-based accessibility

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Attaullah-Kahtana/AI-Based-hERG-Effect-Prediction.git
```

Move into the project directory:

```bash
cd AI-Based-hERG-Effect-Prediction
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ☁️ Hugging Face Deployment

The deployment version requires the application and the trained model.

Recommended Space structure:

```text
AI-Based-hERG-Prediction/
│
├── app.py
├── requirements.txt
└── models/
    └── random_forest_tuned.pkl
```

The trained model is loaded by the application:

```python
joblib.load("models/random_forest_tuned.pkl")
```

The model does **not** need to be retrained every time a user opens the application.

---


# 🧪 Example

Input:

```text
CCO
```

The application processes the structure and generates molecular descriptors before passing them to the trained model.

Output:

```text
Predicted hERG Effect:
Non-Blocker
```

The prediction is computational and should be experimentally validated before making scientific or pharmaceutical decisions.

---

# ⚠️ Disclaimer

This project is intended for **research, educational, and computational screening purposes only**.

The predictions generated by this model:

* are not clinical diagnoses;
* are not experimental measurements;
* should not be considered definitive evidence of hERG activity;
* should not be used alone for drug-safety or clinical decisions.

Experimental validation and appropriate pharmacological/toxicological assessment are required for scientific conclusions.

---


# 👨‍🔬 Author

## Attaullah

**Bioinformatics | Computational Drug Discovery | Machine Learning | AI**

GitHub:
https://github.com/Attaullah-Kahtana

LinkedIn:
https://www.linkedin.com/in/atta-ullah-67a310331/

---


## ⭐ Project Goal

The primary goal of this project is to demonstrate how **cheminformatics, machine learning, explainable AI, and web deployment** can be combined to develop a computational workflow for early-stage **hERG-related cardiotoxicity screening**.

If you find this project useful, consider ⭐ starring the repository and connecting with me on GitHub and LinkedIn.
