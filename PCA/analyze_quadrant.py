import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = "PCA_output/quadrant_pca_recomputed"  # יש לך תתי-תיקיות לכל רבע
OUT_DIR = "PCA_output/quadrant_pca_recomputed"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# ----------------------------
# Helper: recompute PCA per quadrant
# ----------------------------
def recompute_pca_for_csv(csv_path: str):
    q_name = os.path.basename(csv_path).replace(".csv", "")
    df = pd.read_csv(csv_path)

    if "dream" not in df.columns:
        print(f"⚠️ Skipping {q_name}: no 'dream' column")
        return None

    texts = df["dream"].astype(str).tolist()
    if len(texts) < 10:
        print(f"⚠️ Skipping {q_name}: not enough rows ({len(texts)})")
        return None

    # embeddings per quadrant
    emb = model.encode(texts, show_progress_bar=True)

    # safety: avoid NaN/inf (מונע warnings כמו overflow/invalid)
    emb = np.asarray(emb, dtype=np.float32)
    emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)

    # PCA per quadrant
    pca = PCA(n_components=2, random_state=42, svd_solver="randomized")
    coords = pca.fit_transform(emb)
    x, y = coords[:, 0], coords[:, 1]

    # output folder per quadrant
    q_out = os.path.join(OUT_DIR, q_name)
    os.makedirs(q_out, exist_ok=True)

    # save CSV with new local PCA coords
    df["q_pca1"] = x
    df["q_pca2"] = y
    df.to_csv(os.path.join(q_out, f"{q_name}_with_quadrant_pca.csv"), index=False)

    # plot
    plt.figure(figsize=(7.5, 6))
    plt.scatter(x, y, s=6, alpha=0.6)
    plt.xlabel("Quadrant PCA1")
    plt.ylabel("Quadrant PCA2")
    evr = pca.explained_variance_ratio_
    plt.title(f"{q_name}\nEVR: {evr[0]:.2%}, {evr[1]:.2%}")
    plt.tight_layout()

    img_path = os.path.join(q_out, "pca_scatter.png")
    plt.savefig(img_path, dpi=250)
    plt.close()

    print(f"✔ Saved: {img_path}")
    return q_name, img_path

# ----------------------------
# 1) Find quadrant CSVs automatically (subfolders)
# ----------------------------
csv_paths = sorted(glob.glob(os.path.join(BASE_DIR, "*", "*.csv")))
if not csv_paths:
    raise FileNotFoundError(f"No CSVs found under: {BASE_DIR}/*/*.csv")

results = []
for csv_path in csv_paths:
    r = recompute_pca_for_csv(csv_path)
    if r:
        results.append(r)

# ----------------------------
# 2) Combine the 4 images into one (2x2)
# ----------------------------
# order by name for consistency
results.sort(key=lambda t: t[0])
img_paths = [p for _, p in results][:4]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for ax, (q_name, img_path) in zip(axes, results[:4]):
    img = plt.imread(img_path)
    ax.imshow(img)
    ax.set_title(q_name)
    ax.axis("off")

plt.tight_layout()
combined_path = os.path.join(OUT_DIR, "all_quadrants_pca.png")
plt.savefig(combined_path, dpi=250)
plt.show()
plt.close()

print(f"✅ Combined image saved: {combined_path}")
