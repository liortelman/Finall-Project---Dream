"""
Axis-orientation check for the GPT-4o predictions (no API calls).
PCA signs are arbitrary. For every possible orientation of the two axes,
recompute GPT-4o's accuracy and per-axis agreement from the SAVED predictions.
"""
import glob
import pandas as pd

FLIP_PCA2 = {"Everyday + Cognitive": "Everyday + Experiential",
             "Everyday + Experiential": "Everyday + Cognitive",
             "Dramatic + Cognitive": "Dramatic + Experiential",
             "Dramatic + Experiential": "Dramatic + Cognitive"}
FLIP_PCA1 = {"Everyday + Cognitive": "Dramatic + Cognitive",
             "Dramatic + Cognitive": "Everyday + Cognitive",
             "Everyday + Experiential": "Dramatic + Experiential",
             "Dramatic + Experiential": "Everyday + Experiential"}

rows = []
for path in sorted(glob.glob("table1_outputs/LLM (gpt-4o)*_predictions.csv")):
    d = pd.read_csv(path)
    pred = d["predicted_quadrant"]
    print(f"\n{path}  (n={len(d)})")
    for name, truth in [("original", d["quadrant"]),
                        ("PCA2 flipped", d["quadrant"].map(FLIP_PCA2)),
                        ("PCA1 flipped", d["quadrant"].map(FLIP_PCA1)),
                        ("both flipped", d["quadrant"].map(FLIP_PCA1).map(FLIP_PCA2))]:
        acc = (pred == truth).mean()
        ax1 = (pred.str.split(" \\+ ").str[0] == truth.str.split(" \\+ ").str[0]).mean()
        ax2 = (pred.str.split(" \\+ ").str[1] == truth.str.split(" \\+ ").str[1]).mean()
        print(f"  {name:13s} acc={acc:.3f}  axis1={ax1:.3f}  axis2={ax2:.3f}")
        rows.append({"file": path, "n": len(d), "orientation": name,
                     "accuracy": round(acc, 4),
                     "axis1_agreement": round(ax1, 4),
                     "axis2_agreement": round(ax2, 4)})

pd.DataFrame(rows).to_csv("table1_outputs/axis_orientation_check.csv", index=False)
print("\nSaved: table1_outputs/axis_orientation_check.csv")