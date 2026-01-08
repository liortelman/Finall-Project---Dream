import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# ===============================
# Configuration
# ===============================
BASE_DIR = "PCA_output/quadrants"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# צבעים זהים לגרף הראשי
QUADRANT_COLORS = {
    "dramatic_plus_experiential": "crimson",
    "everyday_plus_experiential": "orange",
    "dramatic_plus_cognitive": "royalblue",
    "everyday_plus_cognitive": "seagreen",
}

# כותרות יפות
QUADRANT_TITLES = {
    "dramatic_plus_experiential": "Dramatic + Experiential",
    "everyday_plus_experiential": "Everyday + Experiential",
    "dramatic_plus_cognitive": "Dramatic + Cognitive",
    "everyday_plus_cognitive": "Everyday + Cognitive",
}

# ===============================
# PCA per quadrant (שומר בתיקייה הקיימת)
# ===============================
def run_quadrant_pca(csv_path: str):
    quadrant_dir = os.path.dirname(csv_path)
    quadrant_name = os.path.basename(csv_path).replace(".csv", "")
    color = QUADRANT_COLORS.get(quadrant_name, "gray")

    df = pd.read_csv(csv_path)
    if "dream" not in df.columns:
        print(f"⚠ Skipping {quadrant_name}: no 'dream' column")
        return None

    texts = df["dream"].astype(str).tolist()
    if len(texts) < 10:
        print(f"⚠ Skipping {quadrant_name}: too few samples")
        return None

    # embeddings
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    embeddings = np.nan_to_num(embeddings)

    # PCA
    pca = PCA(n_components=2, random_state=42, svd_solver="randomized")
    coords = pca.fit_transform(embeddings)
    x, y = coords[:, 0], coords[:, 1]

    # save CSV (באותה תיקייה)
    df["q_pca1"] = x
    df["q_pca2"] = y
    out_csv = os.path.join(
        quadrant_dir, f"{quadrant_name}_with_quadrant_pca.csv"
    )
    df.to_csv(out_csv, index=False)

    # plot (בלי X)
    plt.figure(figsize=(7, 6))
    plt.scatter(x, y, s=6, alpha=0.6, color=color)

    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(0, color="black", linewidth=0.8)

    evr = pca.explained_variance_ratio_
    plt.xlabel("Local PCA1 — semantic variation within this quadrant")
    plt.ylabel("Local PCA2 — semantic variation within this quadrant")
    plt.title(
        f"{QUADRANT_TITLES.get(quadrant_name, quadrant_name)}\n"
        f"Explained variance: PC1={evr[0]:.2%}, PC2={evr[1]:.2%}"
    )

    plt.tight_layout()
    img_path = os.path.join(quadrant_dir, "pca_scatter.png")
    plt.savefig(img_path, dpi=300)
    plt.close()

    print(f"✔ Finished {quadrant_name}")
    return quadrant_name, img_path

# ===============================
# Run PCA on all quadrants
# ===============================
csv_files = glob.glob(os.path.join(BASE_DIR, "*", "*.csv"))
results = []

for csv_path in sorted(csv_files):
    res = run_quadrant_pca(csv_path)
    if res:
        results.append(res)

# ===============================
# Combine into one 2x2 image (בתיקיית quadrants)
# ===============================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, (q_name, img_path) in zip(axes, results):
    img = plt.imread(img_path)
    ax.imshow(img)
    ax.set_title(QUADRANT_TITLES.get(q_name, q_name))
    ax.axis("off")

plt.suptitle(
    "Local PCA Structure Within Each Dream Quadrant",
    fontsize=16
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
combined_path = os.path.join(BASE_DIR, "all_quadrants_pca.png")
plt.savefig(combined_path, dpi=300)
plt.show()
plt.close()

print("✅ Combined image saved to:", combined_path)
