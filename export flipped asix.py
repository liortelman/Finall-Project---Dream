"""
Axis-orientation check for the GPT-4o predictions (no API calls).
PCA signs are arbitrary. For every possible orientation of the two axes,
recompute GPT-4o's accuracy and per-axis agreement from the SAVED predictions,
and export Table 2 (CSV, Markdown and LaTeX).
"""
import glob
import os
import sys

import pandas as pd

OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "table1_outputs"

FLIP_PCA2 = {"Everyday + Cognitive": "Everyday + Experiential",
             "Everyday + Experiential": "Everyday + Cognitive",
             "Dramatic + Cognitive": "Dramatic + Experiential",
             "Dramatic + Experiential": "Dramatic + Cognitive"}
FLIP_PCA1 = {"Everyday + Cognitive": "Dramatic + Cognitive",
             "Dramatic + Cognitive": "Everyday + Cognitive",
             "Everyday + Experiential": "Dramatic + Experiential",
             "Dramatic + Experiential": "Everyday + Experiential"}

ORIENTATIONS = [
    ("Original (as in Table 1)", lambda q: q),
    ("PCA2 flipped",             lambda q: q.map(FLIP_PCA2)),
    ("PCA1 flipped",             lambda q: q.map(FLIP_PCA1)),
    ("Both flipped",             lambda q: q.map(FLIP_PCA1).map(FLIP_PCA2)),
]


def axis(series, i):
    return series.str.split(" \\+ ").str[i]


long_rows = []
table = {name: {} for name, _ in ORIENTATIONS}
n_test = None

paths = sorted(glob.glob(os.path.join(OUTPUT_DIR, "LLM (gpt-4o)*_predictions.csv")))
if not paths:
    sys.exit(f"No GPT-4o prediction files found in {OUTPUT_DIR}")

for path in paths:
    mode = "CoT" if "cot" in os.path.basename(path).lower() else "Zero-Shot"
    d = pd.read_csv(path)
    n_test = len(d)
    pred = d["predicted_quadrant"]
    print(f"\n{path}  ({mode}, n={len(d)})")
    for name, transform in ORIENTATIONS:
        truth = transform(d["quadrant"])
        acc = (pred == truth).mean()
        ax1 = (axis(pred, 0) == axis(truth, 0)).mean()
        ax2 = (axis(pred, 1) == axis(truth, 1)).mean()
        print(f"  {name:25s} acc={acc:.3f}  axis1={ax1:.3f}  axis2={ax2:.3f}")
        long_rows.append({"file": path, "setup": mode, "n": len(d), "orientation": name,
                          "accuracy": round(acc, 4),
                          "axis1_agreement": round(ax1, 4),
                          "axis2_agreement": round(ax2, 4)})
        table[name][f"{mode} Acc."] = acc
        table[name][f"{mode} Axis 1"] = ax1
        table[name][f"{mode} Axis 2"] = ax2

# Detailed (long) results
pd.DataFrame(long_rows).to_csv(os.path.join(OUTPUT_DIR, "axis_orientation_check.csv"), index=False)

# Table 2 (wide, as in the report)
columns = [c for c in ["Zero-Shot Acc.", "Zero-Shot Axis 1", "Zero-Shot Axis 2",
                       "CoT Acc.", "CoT Axis 1", "CoT Axis 2"]
           if any(c in row for row in table.values())]
table2 = pd.DataFrame.from_dict(table, orient="index")[columns].round(2)
table2.index.name = "Orientation"

table2.to_csv(os.path.join(OUTPUT_DIR, "table2_axis_orientation.csv"))
try:  # needs the "tabulate" package
    with open(os.path.join(OUTPUT_DIR, "table2_axis_orientation.md"), "w") as f:
        f.write(table2.to_markdown(floatfmt=".2f"))
        f.write(f"\n\nTable 2: GPT-4o accuracy and per-axis agreement under different "
                f"PCA axis orientations (n = {n_test}).\n")
except ImportError:
    print("Skipping .md export (pip install tabulate)")
try:  # needs the "jinja2" package on newer pandas
    table2.to_latex(os.path.join(OUTPUT_DIR, "table2_axis_orientation.tex"), float_format="%.2f",
                    caption=f"GPT-4o accuracy and per-axis agreement under different "
                            f"PCA axis orientations (n = {n_test}).",
                    label="tab:axis_orientation")
except ImportError:
    print("Skipping .tex export (pip install jinja2)")

print("\n" + "=" * 70)
print(f"TABLE 2 (n={n_test})")
print("=" * 70)
print(table2.to_string(float_format="%.2f"))
print(f"\nSaved to {OUTPUT_DIR}/: axis_orientation_check.csv, table2_axis_orientation.csv/.md/.tex")

# ===============================
# Evidence for PCA2 orientation (Results, item 5):
# Spearman correlation between the PCA coordinates and report length (training set)
# ===============================
from scipy.stats import spearmanr

train = pd.read_csv(os.path.join(OUTPUT_DIR, "train.csv"))
length = train["dream"].astype(str).str.split().str.len()
rho_pca2 = spearmanr(train["pca2"], length)[0]
rho_pca1 = spearmanr(train["pca1"], length)[0]


print(f"\nSpearman rho with report length (train, n={len(train)}): "
      f"PCA2 = {rho_pca2:.2f}, PCA1 = {rho_pca1:.2f}")
pd.DataFrame([{"split": "train", "n": len(train),
               "spearman_pca2_vs_length": round(rho_pca2, 4),
               "spearman_pca1_vs_length": round(rho_pca1, 4)}]).to_csv(
    os.path.join(OUTPUT_DIR, "pca_length_correlation.csv"), index=False)
print(f"Saved to {OUTPUT_DIR}/pca_length_correlation.csv")