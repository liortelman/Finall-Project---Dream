"""
extract_for_game.py
-------------------
מריץ PCA על החלומות ומייצר קובץ JSON מוכן למשחק הוולידציה.

הרץ:  python extract_for_game.py
פלט:  game_data.json  (בתיקייה הנוכחית)
"""

import random
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# ── הגדרות ──────────────────────────────────────────────────
CSV_PATH      = "all_dreams_combined.csv"   # שנה אם הקובץ במקום אחר
N_COMPONENTS  = 5       # מספר צירי PCA
N_SAMPLE      = 5000    # גודל מדגם (יותר = מדויק יותר, אבל איטי יותר)
N_EXTREME     = 20      # כמה חלומות לשמור בכל קצה של כל ציר
RANDOM_SEED   = 42
MAX_DREAM_LEN = 600     # גוזר חלומות ארוכים מדי (תווים)
# ────────────────────────────────────────────────────────────

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("טוען נתונים...")
df = pd.read_csv(CSV_PATH)
dreams_all = df["dream"].astype(str).tolist()

# דגימה אקראית
indices = random.sample(range(len(dreams_all)), min(N_SAMPLE, len(dreams_all)))
dreams  = [dreams_all[i] for i in indices]
print(f"  {len(dreams)} חלומות נבחרו")

print("מחשב embeddings (לוקח כ-2-5 דקות)...")
model      = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(dreams, show_progress_bar=True, batch_size=64)

print(f"מריץ PCA עם {N_COMPONENTS} רכיבים...")
pca    = PCA(n_components=N_COMPONENTS, random_state=RANDOM_SEED)
coords = pca.fit_transform(embeddings)

evr = pca.explained_variance_ratio_
print("Explained variance:", [round(float(v), 4) for v in evr])

# ── בנה את מבנה הנתונים למשחק ──────────────────────────────
def trim(text, max_len=MAX_DREAM_LEN):
    text = text.strip()
    return text[:max_len] + "…" if len(text) > max_len else text

axes = []
for i in range(N_COMPONENTS):
    scores  = coords[:, i]
    sorted_idx = np.argsort(scores)

    positive_idx = sorted_idx[-N_EXTREME:][::-1].tolist()  # גבוהים ביותר
    negative_idx = sorted_idx[:N_EXTREME].tolist()          # נמוכים ביותר

    axes.append({
        "axis_id"        : i + 1,
        "label"          : f"PC{i+1}",
        "explained_var"  : round(float(evr[i]), 4),
        "positive_pole"  : {
            "description": "",   # ← מלאי ידנית אחרי שתראי את החלומות
            "dreams": [
                {"id": idx, "text": trim(dreams[idx]), "score": round(float(scores[idx]), 4)}
                for idx in positive_idx
            ]
        },
        "negative_pole"  : {
            "description": "",   # ← מלאי ידנית
            "dreams": [
                {"id": idx, "text": trim(dreams[idx]), "score": round(float(scores[idx]), 4)}
                for idx in negative_idx
            ]
        }
    })

output = {
    "metadata": {
        "n_dreams_sampled" : len(dreams),
        "n_components"     : N_COMPONENTS,
        "explained_variance": [round(float(v), 4) for v in evr],
        "model"            : "all-MiniLM-L6-v2"
    },
    "axes": axes
}

with open("game_data.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("\n✅ נשמר: game_data.json")
print("השלבים הבאים:")
print("  1. פתחי את game_data.json")
print("  2. בכל ציר, קראי כמה חלומות מהקצה החיובי והשלילי")
print("  3. מלאי את שדה 'description' בכל קוטב (למשל: 'חלומות דרמטיים-רגשיים')")
print("  4. שלחי את הקובץ הזה חזרה לצ'אט")
