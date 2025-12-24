# ===============================
# Imports
# ===============================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


# ===============================
# Configuration
# ===============================
CSV_PATH = "all_dreams_combined.csv"
OUTPUT_DIR = "PCA_output"
MAX_DREAMS = 38222

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===============================
# 1. Load Data
# ===============================
dreams_df = pd.read_csv(CSV_PATH)
print("PCA plots saved to folder:", OUTPUT_DIR)


# Extract dream texts
dreams = dreams_df["dream"].astype(str).tolist()
dreams_subset = dreams[:MAX_DREAMS]


# ===============================
# 2. Embed Dreams (SentenceTransformer)
# ===============================
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(dreams_subset, show_progress_bar=True)
print("Embeddings shape:", embeddings.shape)

# ===============================
# 3. Dimensionality Reduction (PCA → 2D)
# ===============================
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

x = embeddings_2d[:, 0]  # PCA Component 1
y = embeddings_2d[:, 1]  # PCA Component 2


# ===============================
# 4. Plot 1 — Raw PCA Distribution
# ===============================
plt.figure(figsize=(12, 8))
plt.scatter(x, y, s=10, alpha=0.5, color="blue")

# Annotate a few example dreams
for i in range(0, min(50, len(dreams_subset)), 5):
    plt.annotate(
        dreams_subset[i][:20] + "...",
        (x[i], y[i]),
        fontsize=8
    )

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Semantic Distribution of Dreams (PCA)")
plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/01_pca_raw_distribution.png", dpi=300)
plt.close()


# ===============================
# 5. Binary Classification via PCA1 Median
# ===============================
threshold = np.median(x)

labels = np.where(
    x >= threshold,
    "High emotional / dramatic",
    "Everyday / mundane"
)

plt.figure(figsize=(8, 6))

for label, color in [
    ("High emotional / dramatic", "crimson"),
    ("Everyday / mundane", "steelblue"),
]:
    mask = labels == label
    plt.scatter(
        x[mask],
        y[mask],
        s=8,
        alpha=0.6,
        label=label,
        color=color
    )

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title(
    "Dreams Embedded in 2D Space (PCA)\n"
    "Colored by Emotional Intensity (Derived from PCA1)"
)
plt.legend()
plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/02_pca_emotional_binary.png", dpi=300)
plt.close()


# ===============================
# 6. Four-Quadrant Psychological Interpretation
# ===============================
quadrants = {
    "Dramatic + Experiential": (x > 0) & (y > 0),
    "Everyday + Experiential": (x <= 0) & (y > 0),
    "Dramatic + Cognitive": (x > 0) & (y <= 0),
    "Everyday + Cognitive": (x <= 0) & (y <= 0),
}

colors = {
    "Dramatic + Experiential": "crimson",
    "Everyday + Experiential": "orange",
    "Dramatic + Cognitive": "royalblue",
    "Everyday + Cognitive": "seagreen",
}

plt.figure(figsize=(9, 7))

for label, mask in quadrants.items():
    plt.scatter(
        x[mask],
        y[mask],
        s=8,
        alpha=0.6,
        label=label,
        color=colors[label]
    )

# Zero axes
plt.axhline(0, color="black", linewidth=0.8)
plt.axvline(0, color="black", linewidth=0.8)

plt.xlabel("PCA1 – Emotional Intensity (Dramatic Content)")
plt.ylabel("PCA2 – Cognitive ↔ Experiential–Emotional")
plt.title(
    "Dream Semantic Space (PCA)\n"
    "Four-Quadrant Psychological Interpretation"
)

plt.legend(markerscale=2, fontsize=9)
plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/03_pca_dream_semantic_space.png", dpi=300)
plt.close()


# ===============================
# Done
# ===============================
print("PCA plots saved to folder:", OUTPUT_DIR)
