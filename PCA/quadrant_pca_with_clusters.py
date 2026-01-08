import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ===============================
# Configuration
# ===============================
BASE_DIR = "PCA_output/quadrants"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

K = 4  # מספר קבוצות

QUADRANTS = [
    "dramatic_plus_cognitive",
    "dramatic_plus_experiential",
    "everyday_plus_cognitive",
    "everyday_plus_experiential",
]

QUADRANT_TITLES = {
    "dramatic_plus_experiential": "Dramatic + Experiential",
    "everyday_plus_experiential": "Everyday + Experiential",
    "dramatic_plus_cognitive": "Dramatic + Cognitive",
    "everyday_plus_cognitive": "Everyday + Cognitive",
}

# ===============================
# Run on ONE main file per quadrant: <quadrant>/<quadrant>.csv
# ===============================
def run_for_quadrant(quadrant: str):
    quadrant_dir = os.path.join(BASE_DIR, quadrant)
    main_csv = os.path.join(quadrant_dir, f"{quadrant}.csv")   # <-- הקובץ הראשי בלבד

    if not os.path.exists(main_csv):
        print(f"⚠ Missing main CSV: {main_csv}")
        return None

    df = pd.read_csv(main_csv)
    if "dream" not in df.columns:
        print(f"⚠ Skipping {quadrant}: no 'dream' column")
        return None

    texts = df["dream"].astype(str).tolist()
    n = len(texts)
    if n < K + 1:
        print(f"⚠ Skipping {quadrant}: not enough samples ({n}) for k={K}")
        return None

    # embeddings
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    embeddings = np.nan_to_num(embeddings)

    # local PCA
    pca = PCA(n_components=2, random_state=42, svd_solver="randomized")
    coords = pca.fit_transform(embeddings)
    x, y = coords[:, 0], coords[:, 1]

    # KMeans on embeddings
    kmeans = KMeans(n_clusters=K, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(embeddings)

    # save CSV (same folder)
    out_csv = os.path.join(quadrant_dir, f"{quadrant}_with_quadrant_pca_clusters_k{K}.csv")
    df_out = df.copy()
    df_out["q_pca1"] = x
    df_out["q_pca2"] = y
    df_out["cluster"] = labels
    df_out.to_csv(out_csv, index=False)

    # plot (same folder)
    plt.figure(figsize=(7.5, 6))
    sc = plt.scatter(x, y, c=labels, s=7, alpha=0.65)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(0, color="black", linewidth=0.8)

    evr = pca.explained_variance_ratio_
    plt.xlabel("Local PCA1 — semantic variation within this quadrant")
    plt.ylabel("Local PCA2 — semantic variation within this quadrant")

    title = QUADRANT_TITLES.get(quadrant, quadrant)
    plt.title(
        f"{title}\n"
        f"KMeans clusters (k={K}) | PC1={evr[0]:.2%}, PC2={evr[1]:.2%}"
    )

    cbar = plt.colorbar(sc)
    cbar.set_label("Cluster ID")

    plt.tight_layout()
    out_img = os.path.join(quadrant_dir, f"pca_scatter_clusters_k{K}.png")
    plt.savefig(out_img, dpi=300)
    plt.close()

    # console summary
    counts = pd.Series(labels).value_counts().sort_index().to_dict()
    print(f"✔ {quadrant}: saved {out_img}")
    print("  cluster sizes:", counts)

    return quadrant, out_img

# ===============================
# Run for all 4 quadrants
# ===============================
results = []
for q in QUADRANTS:
    res = run_for_quadrant(q)
    if res:
        results.append(res)

if len(results) != 4:
    print("\n⚠ Not all quadrants were processed. Check missing main CSV paths above.")

# ===============================
# Combine into one 2x2 image (saved in BASE_DIR)
# ===============================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# keep same order
results_by_name = {name: path for name, path in results}

for ax, q in zip(axes, QUADRANTS):
    if q not in results_by_name:
        ax.axis("off")
        ax.set_title(f"{q} (missing)")
        continue

    img = plt.imread(results_by_name[q])
    ax.imshow(img)
    ax.set_title(QUADRANT_TITLES.get(q, q))
    ax.axis("off")

plt.suptitle(f"Local PCA + KMeans Clusters (k={K}) — One Main File Per Quadrant", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])

combined_path = os.path.join(BASE_DIR, f"all_quadrants_pca_clusters_k{K}.png")
plt.savefig(combined_path, dpi=300)
plt.show()
plt.close()

print("\n✅ Combined image saved to:", combined_path)
