import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# Configuration
# ============================================================

# PC1.py נמצא בתוך תיקיית PCA, לכן עולים תיקייה אחת
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# הקובץ שמכיל את כל החלומות
# עדכני את שם הקובץ כאן במידת הצורך
INPUT_CSV = os.path.join(
    PROJECT_DIR,
    "PCA",
    "all_dreams_combined.csv",
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "PCA_output",
    "narrative_axis",
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DREAM_COLUMN = "dream"

# כמה חלומות לבחור מכל צד לפני ה-PCA
# לדוגמה: 500 fragmented + 500 rich narrative
DREAMS_PER_SIDE = 500

# מספר החלומות הקיצוניים שנוציא מכל צד
NUMBER_OF_EXTREMES = 5

# אפשר להגביל את מספר החלומות מהקובץ המלא לצורכי בדיקה.
# None = להשתמש בכל החלומות
MAX_DREAMS = None

RANDOM_STATE = 42
BATCH_SIZE = 64


# ============================================================
# Output files
# ============================================================

SELECTED_GROUP_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "narrative_axis_selected_dreams.csv",
)

ALL_PC1_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "narrative_axis_dreams_with_pc1.csv",
)

LEFT_EXTREMES_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "fragmented_extreme_dreams.csv",
)

RIGHT_EXTREMES_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "rich_narrative_extreme_dreams.csv",
)

GRAPH_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "fragmented_to_rich_narrative_pc1.png",
)


# ============================================================
# Narrative semantic anchors
# ============================================================

FRAGMENTED_ANCHORS = [
    "A fragmented dream with disconnected images and no clear narrative.",
    "A dream made of short unrelated fragments without a storyline.",
    "A dream with isolated scenes and no clear sequence of events.",
    "A vague dream containing disconnected people, objects, and places.",
    "A dream report with incomplete fragments and little narrative structure.",
    "Several unrelated dream images with no beginning, progression, or ending.",
]

RICH_NARRATIVE_ANCHORS = [
    "A detailed dream with a clear sequence of connected events.",
    "A coherent dream narrative with characters, actions, and a storyline.",
    "A rich and detailed story with a beginning, progression, and conclusion.",
    "A dream in which one event clearly leads to the next.",
    "A developed dream narrative describing actions, motivations, and outcomes.",
    "A long coherent dream with connected scenes and a clear plot.",
]


# ============================================================
# Helper functions
# ============================================================

def validate_input_file(input_path: str) -> None:
    """
    Check that the input CSV exists.
    """

    print("Project directory:", PROJECT_DIR)
    print("Looking for input file at:", input_path)

    if not os.path.exists(input_path):
        print("\nCSV files found inside the project:")

        found_csv = False

        for root, _, files in os.walk(PROJECT_DIR):
            for filename in files:
                if filename.lower().endswith(".csv"):
                    found_csv = True
                    print(os.path.join(root, filename))

        if not found_csv:
            print("No CSV files were found.")

        raise FileNotFoundError(
            "\nThe file containing all dreams was not found.\n"
            "Update INPUT_CSV to one of the paths printed above."
        )


def load_dreams(input_path: str) -> pd.DataFrame:
    """
    Load and clean all dreams.
    """

    df = pd.read_csv(input_path)

    if DREAM_COLUMN not in df.columns:
        raise ValueError(
            f"The CSV must contain a column named '{DREAM_COLUMN}'.\n"
            f"Existing columns: {list(df.columns)}"
        )

    df = df.dropna(
        subset=[DREAM_COLUMN]
    ).copy()

    df[DREAM_COLUMN] = (
        df[DREAM_COLUMN]
        .astype(str)
        .str.strip()
    )

    df = df[
        df[DREAM_COLUMN] != ""
    ].copy()

    # Remove identical dream texts
    df = df.drop_duplicates(
        subset=[DREAM_COLUMN]
    ).reset_index(drop=True)

    if MAX_DREAMS is not None:
        df = df.head(MAX_DREAMS).copy()

    if len(df) < NUMBER_OF_EXTREMES * 2:
        raise ValueError(
            "There are not enough valid dreams in the input file."
        )

    return df


def create_embeddings(
    model: SentenceTransformer,
    texts: list[str],
) -> np.ndarray:
    """
    Create normalized sentence embeddings.
    """

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    embeddings = np.nan_to_num(
        embeddings,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return embeddings


def create_anchor_vector(
    model: SentenceTransformer,
    anchors: list[str],
) -> np.ndarray:
    """
    Create one average semantic vector from several anchor sentences.
    """

    anchor_embeddings = model.encode(
        anchors,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    anchor_vector = np.mean(
        anchor_embeddings,
        axis=0,
    )

    vector_norm = np.linalg.norm(anchor_vector)

    if vector_norm > 0:
        anchor_vector = anchor_vector / vector_norm

    return anchor_vector


def calculate_narrative_scores(
    embeddings: np.ndarray,
    fragmented_anchor: np.ndarray,
    rich_anchor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate semantic similarity to both ends of the narrative axis.

    narrative_direction_score:
        negative = closer to fragmented
        positive = closer to rich narrative

    narrative_relevance:
        high value = strongly connected to either end of the axis
    """

    fragmented_similarity = cosine_similarity(
        embeddings,
        fragmented_anchor.reshape(1, -1),
    ).flatten()

    rich_similarity = cosine_similarity(
        embeddings,
        rich_anchor.reshape(1, -1),
    ).flatten()

    narrative_direction_score = (
        rich_similarity - fragmented_similarity
    )

    narrative_relevance = np.abs(
        narrative_direction_score
    )

    return (
        fragmented_similarity,
        rich_similarity,
        narrative_direction_score,
        narrative_relevance,
    )


def select_narrative_axis_group(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select an equal number of dreams from both sides:

    - Most fragmented dreams
    - Richest narrative dreams

    Selection is based on semantic anchor scores.
    """

    possible_per_side = min(
        DREAMS_PER_SIDE,
        len(df) // 2,
    )

    fragmented_group = (
        df
        .nsmallest(
            possible_per_side,
            "narrative_direction_score",
        )
        .copy()
    )

    rich_group = (
        df
        .nlargest(
            possible_per_side,
            "narrative_direction_score",
        )
        .copy()
    )

    fragmented_group["initial_semantic_group"] = (
        "Fragmented / No Narrative"
    )

    rich_group["initial_semantic_group"] = (
        "Rich Narrative / Storyline"
    )

    selected_df = pd.concat(
        [
            fragmented_group,
            rich_group,
        ],
        ignore_index=True,
    )

    # Prevent duplicates in case the requested sample size is very large
    selected_df = selected_df.drop_duplicates(
        subset=[DREAM_COLUMN]
    ).reset_index(drop=True)

    return selected_df


def orient_pc1(
    pc1: np.ndarray,
    direction_scores: np.ndarray,
) -> tuple[np.ndarray, bool, float]:
    """
    PCA axis signs are arbitrary.

    Ensure that:
        low PC1  = fragmented
        high PC1 = rich narrative
    """

    correlation = np.corrcoef(
        pc1,
        direction_scores,
    )[0, 1]

    if np.isnan(correlation):
        correlation = 0.0

    if correlation < 0:
        return -pc1, True, correlation

    return pc1, False, correlation


# ============================================================
# Start
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

validate_input_file(INPUT_CSV)

print("\nLoading all dreams...")

all_dreams_df = load_dreams(
    INPUT_CSV
)

print(
    f"Valid unique dreams: {len(all_dreams_df):,}"
)


# ============================================================
# Load model
# ============================================================

print(f"\nLoading model: {MODEL_NAME}")

model = SentenceTransformer(
    MODEL_NAME
)


# ============================================================
# Embed all dreams
# ============================================================

print("\nCreating embeddings for all dreams...")

all_embeddings = create_embeddings(
    model,
    all_dreams_df[DREAM_COLUMN].tolist(),
)


# ============================================================
# Create narrative anchors
# ============================================================

print("\nCreating semantic narrative anchors...")

fragmented_anchor = create_anchor_vector(
    model,
    FRAGMENTED_ANCHORS,
)

rich_anchor = create_anchor_vector(
    model,
    RICH_NARRATIVE_ANCHORS,
)


# ============================================================
# Score all dreams
# ============================================================

(
    fragmented_similarity,
    rich_similarity,
    narrative_direction_score,
    narrative_relevance,
) = calculate_narrative_scores(
    embeddings=all_embeddings,
    fragmented_anchor=fragmented_anchor,
    rich_anchor=rich_anchor,
)

scored_df = all_dreams_df.copy()

scored_df["fragmented_similarity"] = (
    fragmented_similarity
)

scored_df["rich_narrative_similarity"] = (
    rich_similarity
)

scored_df["narrative_direction_score"] = (
    narrative_direction_score
)

scored_df["narrative_relevance"] = (
    narrative_relevance
)


# Keep an index connecting the DataFrame to the embedding matrix
scored_df["embedding_index"] = np.arange(
    len(scored_df)
)


# ============================================================
# Select narrative-related dream group
# ============================================================

selected_df = select_narrative_axis_group(
    scored_df
)

print(
    f"\nSelected narrative-axis dreams: "
    f"{len(selected_df):,}"
)

print(
    selected_df["initial_semantic_group"]
    .value_counts()
)


# Save selected group before PCA
selected_df.to_csv(
    SELECTED_GROUP_OUTPUT,
    index=False,
)


# Obtain embeddings of selected dreams
selected_embedding_indices = (
    selected_df["embedding_index"]
    .astype(int)
    .to_numpy()
)

selected_embeddings = all_embeddings[
    selected_embedding_indices
]


# ============================================================
# PCA — one component only
# ============================================================

print("\nRunning PCA with one component...")

pca = PCA(
    n_components=1,
    random_state=RANDOM_STATE,
    svd_solver="randomized",
)

pc1 = pca.fit_transform(
    selected_embeddings
).flatten()

explained_variance = (
    pca.explained_variance_ratio_[0]
)

print(
    f"PC1 explained variance: "
    f"{explained_variance:.2%}"
)


# ============================================================
# Orient PC1
# ============================================================

pc1, axis_flipped, original_correlation = orient_pc1(
    pc1=pc1,
    direction_scores=selected_df[
        "narrative_direction_score"
    ].to_numpy(),
)

final_correlation = np.corrcoef(
    pc1,
    selected_df[
        "narrative_direction_score"
    ].to_numpy(),
)[0, 1]

print(
    "Original correlation between PC1 and narrative score: "
    f"{original_correlation:.4f}"
)

print(
    "Final correlation between PC1 and narrative score: "
    f"{final_correlation:.4f}"
)

print(
    "PC1 axis flipped:",
    axis_flipped,
)


# ============================================================
# Add PC1 results
# ============================================================

selected_df["pc1"] = pc1

selected_df["pc1_side"] = np.where(
    selected_df["pc1"] < 0,
    "Fragmented / No Narrative",
    "Rich Narrative / Storyline",
)

selected_df["pc1_axis_flipped"] = axis_flipped

selected_df["pc1_explained_variance"] = (
    explained_variance
)

selected_df = selected_df.sort_values(
    by="pc1",
    ascending=True,
).reset_index(drop=True)

selected_df.to_csv(
    ALL_PC1_OUTPUT,
    index=False,
)


# ============================================================
# Extract five extreme dreams from each side
# ============================================================

left_extremes = (
    selected_df
    .nsmallest(
        NUMBER_OF_EXTREMES,
        "pc1",
    )
    .copy()
)

right_extremes = (
    selected_df
    .nlargest(
        NUMBER_OF_EXTREMES,
        "pc1",
    )
    .sort_values(
        by="pc1",
        ascending=False,
    )
    .copy()
)

left_extremes["extreme_side"] = (
    "Fragmented / No Narrative"
)

right_extremes["extreme_side"] = (
    "Rich Narrative / Storyline"
)

left_extremes.to_csv(
    LEFT_EXTREMES_OUTPUT,
    index=False,
)

right_extremes.to_csv(
    RIGHT_EXTREMES_OUTPUT,
    index=False,
)


# ============================================================
# Print extreme dreams
# ============================================================

print("\n" + "=" * 90)
print("5 EXTREME DREAMS — LEFT")
print("FRAGMENTED / NO NARRATIVE")
print("=" * 90)

for rank, (_, row) in enumerate(
    left_extremes.iterrows(),
    start=1,
):
    print(
        f"\nL{rank} | PC1 = {row['pc1']:.4f}"
    )

    print(
        f"Anchor score = "
        f"{row['narrative_direction_score']:.4f}"
    )

    print(row[DREAM_COLUMN])


print("\n" + "=" * 90)
print("5 EXTREME DREAMS — RIGHT")
print("RICH NARRATIVE / STORYLINE")
print("=" * 90)

for rank, (_, row) in enumerate(
    right_extremes.iterrows(),
    start=1,
):
    print(
        f"\nR{rank} | PC1 = {row['pc1']:.4f}"
    )

    print(
        f"Anchor score = "
        f"{row['narrative_direction_score']:.4f}"
    )

    print(row[DREAM_COLUMN])


# ============================================================
# One-dimensional PC1 plot
# ============================================================

rng = np.random.default_rng(
    RANDOM_STATE
)

# Jitter is only visual.
# The vertical position has no analytical meaning.
y_jitter = rng.normal(
    loc=0.0,
    scale=0.025,
    size=len(selected_df),
)

fig, ax = plt.subplots(
    figsize=(16, 5),
)


# Background sections
ax.axvspan(
    selected_df["pc1"].min(),
    0,
    alpha=0.06,
)

ax.axvspan(
    0,
    selected_df["pc1"].max(),
    alpha=0.06,
)


# All selected dreams
ax.scatter(
    selected_df["pc1"],
    y_jitter,
    s=18,
    alpha=0.42,
    edgecolors="none",
    label="Selected dreams",
)


# Locate extreme rows after sorting
left_mask = selected_df["pc1"].isin(
    left_extremes["pc1"]
)

right_mask = selected_df["pc1"].isin(
    right_extremes["pc1"]
)


# Highlight left extremes
ax.scatter(
    selected_df.loc[left_mask, "pc1"],
    y_jitter[left_mask.to_numpy()],
    s=90,
    marker="o",
    edgecolors="black",
    linewidths=1,
    zorder=4,
    label="5 most fragmented",
)


# Highlight right extremes
ax.scatter(
    selected_df.loc[right_mask, "pc1"],
    y_jitter[right_mask.to_numpy()],
    s=90,
    marker="o",
    edgecolors="black",
    linewidths=1,
    zorder=4,
    label="5 richest narratives",
)


# Label left extremes
for rank, (_, row) in enumerate(
    left_extremes.iterrows(),
    start=1,
):
    ax.annotate(
        f"L{rank}",
        xy=(row["pc1"], 0),
        xytext=(0, 22),
        textcoords="offset points",
        ha="center",
        fontsize=10,
        fontweight="bold",
    )


# Label right extremes
for rank, (_, row) in enumerate(
    right_extremes.iterrows(),
    start=1,
):
    ax.annotate(
        f"R{rank}",
        xy=(row["pc1"], 0),
        xytext=(0, 22),
        textcoords="offset points",
        ha="center",
        fontsize=10,
        fontweight="bold",
    )


# Center
ax.axvline(
    x=0,
    color="black",
    linewidth=1.1,
    alpha=0.75,
)


# End labels
ax.text(
    0.01,
    0.92,
    "FRAGMENTED / NO NARRATIVE",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=13,
    fontweight="bold",
)

ax.text(
    0.99,
    0.92,
    "RICH NARRATIVE / STORYLINE",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=13,
    fontweight="bold",
)


# Formatting
ax.set_xlabel(
    "PC1: Fragmented / No Narrative  →  Rich Narrative / Storyline",
    fontsize=13,
    labelpad=14,
)

ax.set_title(
    "Dreams Along the Narrative Structure Axis\n"
    f"PC1 explained variance: {explained_variance:.2%}",
    fontsize=16,
    pad=18,
)

ax.set_yticks([])
ax.set_ylabel("")

ax.grid(
    axis="x",
    linewidth=0.4,
    alpha=0.3,
)

ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, -0.33),
    ncol=3,
)

fig.tight_layout()

fig.savefig(
    GRAPH_OUTPUT,
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close(fig)


# ============================================================
# Final output summary
# ============================================================

print("\n" + "=" * 90)
print("FILES SAVED")
print("=" * 90)

print(
    "Selected narrative group:\n",
    SELECTED_GROUP_OUTPUT,
)

print(
    "\nSelected dreams with PC1:\n",
    ALL_PC1_OUTPUT,
)

print(
    "\nFive fragmented extremes:\n",
    LEFT_EXTREMES_OUTPUT,
)

print(
    "\nFive rich-narrative extremes:\n",
    RIGHT_EXTREMES_OUTPUT,
)

print(
    "\nPC1 graph:\n",
    GRAPH_OUTPUT,
)