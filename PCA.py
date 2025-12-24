import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. Read the CSV
dreams_df = pd.read_csv('all_dreams_combined.csv')

# 2. Extract the dreams column correctly (using dreams_df, not df)
# If 'dream' is the column name:
dreams = dreams_df["dream"].astype(str).tolist()
# If you want the first column regardless of name:
# dreams = dreams_df.iloc[:, 0].astype(str).tolist()

# 3. Process with Model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
dreams_subset = dreams[:38222]
embeddings = model.encode(dreams_subset, show_progress_bar=True)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Initialize PCA to reduce to 2 dimensions
pca = PCA(n_components=2)

# Transform your high-dimensional embeddings into 2D points
embeddings_2d = pca.fit_transform(embeddings)

# Now your existing code will work:
x = embeddings_2d[:, 0]
y = embeddings_2d[:, 1]

plt.figure(figsize=(12, 8))
plt.scatter(x, y, s=10, alpha=0.5, color='blue')

# Optional: Label a few points to see what they represent
for i in range(0, 50, 5):  # Label every 5th dream in the first 50
    plt.annotate(dreams_subset[i][:20] + "...", (x[i], y[i]), fontsize=8)

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Semantic Distribution of Dreams")
plt.show()