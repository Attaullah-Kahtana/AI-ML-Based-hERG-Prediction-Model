

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(r"D:\hERG_Prediction_Model")

COMPARISON_FILE = BASE / "results" / "model_comparison.csv"
RESULTS = BASE / "results"

if not COMPARISON_FILE.exists():
    raise FileNotFoundError(
        f"{COMPARISON_FILE} not found. Run scripts 06-09 first -- "
        "each one appends its own row to this file."
    )

df = pd.read_csv(COMPARISON_FILE)

df_sorted = df.sort_values("ROC_AUC", ascending=False)

print("=" * 60)
print("MODEL COMPARISON (sorted by ROC-AUC)")
print("=" * 60)
print(df_sorted.to_string(index=False))

best_model = df_sorted.iloc[0]["Model"]
print(f"\nBest model by ROC-AUC: {best_model}")



# PLOT

metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC", "MCC"]
metrics_to_plot = [m for m in metrics_to_plot if m in df.columns]

plot_df = df.set_index("Model")[metrics_to_plot]

ax = plot_df.plot(kind="bar", figsize=(11, 6))
ax.set_title("Model Comparison Across Metrics")
ax.set_ylabel("Score")
ax.set_xlabel("Model")
ax.legend(loc="lower right", ncol=2)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(RESULTS / "model_comparison.png", dpi=300)
plt.show()

print("\nSaved:", RESULTS / "model_comparison.png")