"""
4-quadrant dream classification - produces all rows of Table 1.

This script was written after the report was submitted. It runs the
supervised experiment described in the report; the numbers it prints are
the outcome of this run and are expected to differ from the submitted Table 1.

Pipeline:
    Dream texts (cleaned, exact duplicates removed)
        -> SentenceTransformer embeddings
        -> PCA (2 dimensions)
        -> 4 quadrant labels
        -> extreme-case filtering (within each quadrant) + class balancing
        -> 70/15/15 train/validation/test split
        -> BERT / RoBERTa fine-tuning                       (rows 1-2)
        -> GPT-4o Zero-Shot and Zero-Shot + Chain-of-Thought (rows 3-4)
        -> Accuracy / Macro Precision / Macro Recall / Macro F1
        -> table1_outputs/table1_results.csv + table1.tex

Run:
    pip install torch transformers datasets accelerate sentence-transformers scikit-learn pandas openai
    export OPENAI_API_KEY=sk-...      # without it the GPT-4o rows are skipped (never invented)
    python experiments.py

Finished parts are cached in table1_outputs/: a model that already has
<name>_metrics.json is not retrained, and LLM answers are stored in
<name>_responses.jsonl, so an interrupted run continues where it stopped.
Set FORCE_RERUN=1 to ignore the cache.
"""

import inspect
import json
import os
import random
import re
import numpy as np
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


# ============================================================
# Configuration
# ============================================================

SEED = 42

# Path relative to this file, so the script runs on any machine / Colab.
# Override with:  DREAMS_CSV=/path/to/all_dreams_combined.csv python experiments.py
DATA_PATH = os.environ.get(
    "DREAMS_CSV",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "PCA",
        "all_dreams_combined.csv",
    ),
)

# Change this if the dream text column in the repository has
# a different name.
TEXT_COLUMN = "dream"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

MODELS = {
    "BERT-base-uncased": "bert-base-uncased",
    "RoBERTa-base": "roberta-base",
}

LEARNING_RATE = 2e-5
BATCH_SIZE = 16
NUM_EPOCHS = 5
WARMUP_RATIO = 0.1      # report: "AdamW with linear warmup"
MAX_LENGTH = 256        # 512 is ~3x slower; dreams average ~100 words
MIN_WORDS = 10

# "Extreme cases" are points sufficiently far from the PCA axes.
#
# The report says that extreme cases were used, but does not provide
# an exact numerical threshold. We therefore expose the percentile
# explicitly instead of hiding an arbitrary choice.
#
# Within each quadrant, keep dreams whose |PCA1| AND |PCA2| are both
# above that quadrant's 50th percentile (far from both axes).
EXTREME_PERCENTILE = 50

# Downsample every class to the size of the smallest one.
BALANCE_CLASSES = True

OUTPUT_DIR = "table1_outputs"

FORCE_RERUN = os.environ.get("FORCE_RERUN") == "1"

# ---------------- LLM (GPT-4o) ----------------
LLM_MODEL = "gpt-4o"
LLM_SETUPS = {
    "zero_shot": "Zero-Shot Prompting",
    "cot": "Zero-Shot + Chain-of-Thought",
}
LLM_TEMPERATURE = 0.0
LLM_WORKERS = 8           # parallel API requests
LLM_MAX_CHARS = 4000      # truncate very long dreams in the prompt
# Optional cost control, e.g. LLM_LIMIT=100 python experiments.py
LLM_LIMIT = int(os.environ["LLM_LIMIT"]) if os.environ.get("LLM_LIMIT") else None


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Data loading
# ============================================================

def load_dreams(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if TEXT_COLUMN not in df.columns:
        raise ValueError(
            f"Could not find text column '{TEXT_COLUMN}'. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna(subset=[TEXT_COLUMN]).copy()
    n_raw = len(df)

    df[TEXT_COLUMN] = df[TEXT_COLUMN].map(clean_dream)

    df = df[df[TEXT_COLUMN].str.split().str.len() >= MIN_WORDS]

    # ~40% of the rows are exact copies of the same dream coming from
    # different source files. Without this step identical texts end up
    # in both train and test (data leakage -> inflated scores).
    n_dup = int(df[TEXT_COLUMN].duplicated().sum())
    df = df.drop_duplicates(subset=[TEXT_COLUMN])

    print(f"Loaded {n_raw:,} rows, removed {n_dup:,} duplicates "
          f"-> {len(df):,} dreams.")

    df = df.reset_index(drop=True)
    df["id"] = np.arange(len(df))   # 'n' is not unique across series

    return df


_ID_TAG = re.compile(r"^\s*#\S+\s*")                               # "#051-1 "
_DATE_TAG = re.compile(r"^\(\d{4}-\d{2}-\d{2}(?: \(\d+\))?\)\s*")  # "(2007-08-02 (16)) "


def clean_dream(text: str) -> str:
    """Remove DreamBank metadata tags and normalise whitespace."""
    text = _DATE_TAG.sub("", _ID_TAG.sub("", str(text)))
    return re.sub(r"\s+", " ", text).strip()


# ============================================================
# Embeddings + PCA
# ============================================================

def calculate_pca(df: pd.DataFrame) -> pd.DataFrame:

    print("\nGenerating SentenceTransformer embeddings...")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    embeddings = embedding_model.encode(
        df[TEXT_COLUMN].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print("Running PCA...")

    pca = PCA(n_components=2, random_state=SEED)

    coordinates = pca.fit_transform(embeddings)

    result = df.copy()

    result["pca1"] = coordinates[:, 0]
    result["pca2"] = coordinates[:, 1]

    print(
        "Explained variance:",
        pca.explained_variance_ratio_
    )

    print_axis_extremes(result)

    return result


def print_axis_extremes(df: pd.DataFrame, k: int = 5):
    """
    PCA component signs are arbitrary. Print the most extreme dreams on
    each side of each axis so we can confirm (by reading them) that
    PCA1>0 = Dramatic and PCA2>0 = Experiential before using the names.
    If an axis is flipped, multiply that column by -1 in calculate_pca().
    """
    for col in ("pca1", "pca2"):
        for side, part in (("highest", df.nlargest(k, col)),
                           ("lowest", df.nsmallest(k, col))):
            print(f"\n--- {col} {side} ---")
            for text in part[TEXT_COLUMN]:
                print(" *", text[:150])


# ============================================================
# Four semantic quadrants
# ============================================================

LABEL_NAMES = {
    0: "Everyday + Cognitive",
    1: "Everyday + Experiential",
    2: "Dramatic + Cognitive",
    3: "Dramatic + Experiential",
}


def quadrant_label(pca1: float, pca2: float) -> int:
    """
    Quadrants according to the final report:

    PCA1:
        negative -> Everyday
        positive -> Dramatic

    PCA2:
        negative -> Cognitive
        positive -> Experiential
    """

    if pca1 < 0 and pca2 < 0:
        return 0

    if pca1 < 0 and pca2 >= 0:
        return 1

    if pca1 >= 0 and pca2 < 0:
        return 2

    return 3


def assign_quadrants(df: pd.DataFrame) -> pd.DataFrame:

    result = df.copy()

    result["label"] = [
        quadrant_label(x, y)
        for x, y in zip(
            result["pca1"],
            result["pca2"],
        )
    ]

    result["quadrant"] = result["label"].map(LABEL_NAMES)

    print("\nQuadrant distribution:")
    print(result["quadrant"].value_counts())

    return result


# ============================================================
# Extreme-case filtering
# ============================================================

def filter_extreme_cases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove dreams close to either PCA axis.

    The report says extreme cases fall "well into the upper percentiles
    of each quadrant", so thresholds are computed separately inside each
    quadrant. The percentile is explicit (EXTREME_PERCENTILE).
    """

    parts = []

    print("\nExtreme-case thresholds (per quadrant):")

    for label, group in df.groupby("label"):

        pca1_threshold = np.percentile(
            np.abs(group["pca1"]), EXTREME_PERCENTILE
        )
        pca2_threshold = np.percentile(
            np.abs(group["pca2"]), EXTREME_PERCENTILE
        )

        print(
            f"{LABEL_NAMES[label]:<25} "
            f"|PCA1| >= {pca1_threshold:.4f}  "
            f"|PCA2| >= {pca2_threshold:.4f}"
        )

        parts.append(
            group[
                (np.abs(group["pca1"]) >= pca1_threshold)
                & (np.abs(group["pca2"]) >= pca2_threshold)
            ]
        )

    filtered = pd.concat(parts)

    if BALANCE_CLASSES:
        n = filtered["label"].value_counts().min()
        filtered = pd.concat(
            group.sample(n, random_state=SEED)
            for _, group in filtered.groupby("label")
        )

    print(
        f"Remaining dreams: {len(filtered):,} "
        f"of {len(df):,}"
    )

    print("\nFiltered class distribution:")
    print(filtered["quadrant"].value_counts())

    return filtered.reset_index(drop=True)


# ============================================================
# 70 / 15 / 15 split
# ============================================================

def split_dataset(df: pd.DataFrame):

    # First: 70% train, 30% temporary
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=SEED,
        stratify=df["label"],
    )

    # Then split remaining 30% equally -> 15% validation / 15% test
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df["label"],
    )

    print("\nDataset split:")
    print(f"Train:      {len(train_df):,}")
    print(f"Validation: {len(validation_df):,}")
    print(f"Test:       {len(test_df):,}")

    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ============================================================
# HuggingFace datasets
# ============================================================

def create_hf_dataset(df, tokenizer):

    dataset = Dataset.from_pandas(
        df[[TEXT_COLUMN, "label"]],
        preserve_index=False,
    )

    def tokenize(batch):
        return tokenizer(
            batch[TEXT_COLUMN],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=[TEXT_COLUMN],
    )

    return dataset


# ============================================================
# Evaluation
# ============================================================

def calculate_metrics(y_true, y_pred):

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    return {
        "Accuracy": accuracy,
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": f1,
    }


def hf_compute_metrics(eval_pred):
    """Used by the Trainer to pick the best epoch on the validation set."""
    logits, labels = eval_pred
    m = calculate_metrics(labels, np.argmax(logits, axis=1))
    return {
        "accuracy": m["Accuracy"],
        "macro_f1": m["Macro F1"],
    }


def version_safe_kwargs():
    """
    transformers 4.x vs 5.x API differences:
      * warmup_ratio was folded into warmup_steps (float = ratio) in v5
      * Trainer(tokenizer=...) was renamed to processing_class=...
    """
    targ_params = inspect.signature(TrainingArguments.__init__).parameters
    trainer_params = inspect.signature(Trainer.__init__).parameters

    warmup = (
        {"warmup_ratio": WARMUP_RATIO}
        if "warmup_ratio" in targ_params
        else {"warmup_steps": WARMUP_RATIO}
    )
    tok_key = (
        "processing_class"
        if "processing_class" in trainer_params
        else "tokenizer"
    )
    return warmup, tok_key


# ============================================================
# Train one Transformer
# ============================================================

def train_model(
    display_name,
    model_name,
    train_df,
    validation_df,
    test_df,
):

    print("\n" + "=" * 70)
    print(f"Training {display_name}")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=4,
        id2label=LABEL_NAMES,
        label2id={
            name: label
            for label, name in LABEL_NAMES.items()
        },
    )

    train_dataset = create_hf_dataset(
        train_df,
        tokenizer,
    )

    validation_dataset = create_hf_dataset(
        validation_df,
        tokenizer,
    )

    test_dataset = create_hf_dataset(
        test_df,
        tokenizer,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    model_output_dir = os.path.join(
        OUTPUT_DIR,
        model_name.replace("/", "_"),
    )

    warmup_kwargs, tokenizer_key = version_safe_kwargs()

    training_args = TrainingArguments(
        output_dir=model_output_dir,

        learning_rate=LEARNING_RATE,

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,

        num_train_epochs=NUM_EPOCHS,

        weight_decay=0.01,

        # AdamW (Trainer default) + linear warmup / decay
        lr_scheduler_type="linear",
        **warmup_kwargs,

        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,            # keep disk usage small

        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,

        logging_strategy="epoch",

        fp16=torch.cuda.is_available(),

        report_to="none",

        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        compute_metrics=hf_compute_metrics,
        **{tokenizer_key: tokenizer},
    )

    trainer.train()

    prediction_output = trainer.predict(
        test_dataset
    )

    predictions = np.argmax(
        prediction_output.predictions,
        axis=1,
    )

    y_true = test_df["label"].to_numpy()

    metrics = calculate_metrics(
        y_true,
        predictions,
    )

    print(f"\n{display_name} test results:")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Save individual predictions so results can be audited.
    prediction_df = test_df[
        ["id", TEXT_COLUMN, "pca1", "pca2", "label", "quadrant"]
    ].copy()

    prediction_df["prediction"] = predictions

    prediction_df["predicted_quadrant"] = (
        prediction_df["prediction"].map(LABEL_NAMES)
    )

    prediction_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{display_name}_predictions.csv",
        ),
        index=False,
    )

    return metrics, {"n_test": len(test_df)}


# ============================================================
# LLM: Zero-Shot and Zero-Shot + Chain-of-Thought (GPT-4o)
# ============================================================

LLM_SYSTEM_PROMPT = """You are an expert annotator of dream reports. Each dream belongs to exactly one of four semantic quadrants defined by two axes.

AXIS 1 - Emotional Intensity:
- Dramatic: high-stakes or emotionally intense events - accidents, violence, nightmares, fear, danger, death, life-or-death scenarios.
- Everyday: mundane settings and situations - work, routines, shopping, chores, ordinary social interaction.

AXIS 2 - Cognitive vs. Experiential orientation:
- Experiential: told from inside the experience - direct sensory and bodily sensations, feelings, personal emotional tension.
- Cognitive: detached, objective, procedural or observational recollection - what happened, logistics, plans, events described from a distance.

QUADRANTS:
1. Dramatic + Experiential: nightmares, fear, physical sensations, violence, intense emotional trauma.
2. Everyday + Experiential: personal emotional tension, minor discomfort, interpersonal awkwardness in daily settings.
3. Dramatic + Cognitive: major high-stakes events described from a distance, objectively, or observationally.
4. Everyday + Cognitive: work, organizational routines, administrative tasks, mundane daily logistics."""

LLM_USER_PROMPTS = {
    "zero_shot": (
        "Classify the dream below into one of the 4 quadrants.\n"
        'Respond with a JSON object only, exactly in this form: {{"quadrant": <1|2|3|4>}}\n\n'
        'Dream:\n"""{dream}"""'
    ),
    "cot": (
        "Classify the dream below into one of the 4 quadrants.\n"
        "First reason step by step: decide where the dream falls on Axis 1 "
        "(Dramatic vs. Everyday) and on Axis 2 (Experiential vs. Cognitive), "
        "citing evidence from the text. Then give the quadrant, and extract the "
        "3-5 specific phrases from the text that most influenced your decision "
        "(copied verbatim).\n"
        "Respond with a JSON object only, exactly in this form:\n"
        '{{"reasoning": "<step-by-step reasoning>", "quadrant": <1|2|3|4>, '
        '"key_phrases": ["...", "..."]}}\n\n'
        'Dream:\n"""{dream}"""'
    ),
}

# Prompt quadrant number -> our label id (LABEL_NAMES)
PROMPT_NUMBER_TO_LABEL = {
    1: 3,  # Dramatic + Experiential
    2: 1,  # Everyday + Experiential
    3: 2,  # Dramatic + Cognitive
    4: 0,  # Everyday + Cognitive
}


def parse_llm_answer(raw):
    """
    Returns (label id, parsed json). Unparsable answers -> label -1,
    which is always counted as a wrong prediction (never dropped).
    """
    if raw is None:
        return -1, None

    obj = None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if match:
            try:
                obj = json.loads(match.group(0))
            except json.JSONDecodeError:
                obj = None

    number = obj.get("quadrant") if isinstance(obj, dict) else None
    try:
        number = int(str(number).strip())
    except (TypeError, ValueError):
        return -1, obj

    return PROMPT_NUMBER_TO_LABEL.get(number, -1), obj


def ask_llm(client, mode, dream, retries=5):
    messages = [
        {"role": "system", "content": LLM_SYSTEM_PROMPT},
        {"role": "user",
         "content": LLM_USER_PROMPTS[mode].format(dream=dream[:LLM_MAX_CHARS])},
    ]
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=LLM_TEMPERATURE,
                seed=SEED,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content, response.model
        except Exception as error:   # rate limit / network: back off and retry
            wait = 2 ** attempt
            print(f"  API error ({type(error).__name__}), retry in {wait}s")
            time.sleep(wait)
    return None, None


def run_llm(mode, display_name, test_df):

    print("\n" + "=" * 70)
    print(f"Running {display_name} - {LLM_SETUPS[mode]}")
    print("=" * 70)

    from openai import OpenAI
    client = OpenAI()

    if LLM_LIMIT:
        test_df = test_df.head(LLM_LIMIT)

    cache_path = os.path.join(OUTPUT_DIR, f"{display_name}_responses.jsonl")
    cache = {}
    if os.path.exists(cache_path) and not FORCE_RERUN:
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if record["raw"] is not None:
                    cache[record["id"]] = record

    todo = [
        (int(row.id), getattr(row, TEXT_COLUMN))
        for row in test_df.itertuples()
        if int(row.id) not in cache
    ]
    print(f"{len(test_df)} test dreams, {len(cache)} cached, {len(todo)} to query")

    mode_flag = "w" if FORCE_RERUN else "a"
    with open(cache_path, mode_flag, encoding="utf-8") as out, \
            ThreadPoolExecutor(LLM_WORKERS) as pool:
        futures = {pool.submit(ask_llm, client, mode, text): dream_id
                   for dream_id, text in todo}
        for done, future in enumerate(as_completed(futures), 1):
            raw, served_model = future.result()
            record = {"id": futures[future], "raw": raw, "served_model": served_model}
            cache[record["id"]] = record
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            if done % 50 == 0:
                print(f"  {done}/{len(todo)}")

    predictions, reasoning, key_phrases = [], [], []
    for row in test_df.itertuples():
        label, obj = parse_llm_answer(cache.get(int(row.id), {}).get("raw"))
        predictions.append(label)
        reasoning.append(obj.get("reasoning") if isinstance(obj, dict) else None)
        key_phrases.append(obj.get("key_phrases") if isinstance(obj, dict) else None)

    n_invalid = sum(p == -1 for p in predictions)
    metrics = calculate_metrics(test_df["label"].to_numpy(), predictions)

    print(f"\n{display_name} test results ({n_invalid} unparsable answers counted as errors):")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    prediction_df = test_df[["id", TEXT_COLUMN, "pca1", "pca2", "label", "quadrant"]].copy()
    prediction_df["prediction"] = predictions
    prediction_df["predicted_quadrant"] = prediction_df["prediction"].map(LABEL_NAMES).fillna("INVALID")
    if mode == "cot":
        prediction_df["reasoning"] = reasoning
        prediction_df["key_phrases"] = key_phrases
    prediction_df.to_csv(
        os.path.join(OUTPUT_DIR, f"{display_name}_predictions.csv"),
        index=False,
    )

    served = sorted({r["served_model"] for r in cache.values() if r.get("served_model")})
    return metrics, {"n_test": len(test_df), "n_invalid": n_invalid,
                     "served_model_versions": served}


# ============================================================
# Cached runs
# ============================================================

def cached_run(display_name, setup, run_fn):
    """Run once and store metrics; reuse the stored metrics on later runs."""
    path = os.path.join(OUTPUT_DIR, f"{display_name}_metrics.json")

    if os.path.exists(path) and not FORCE_RERUN:
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
        print(f"\nUsing saved results for {display_name} ({path})")
        return saved["metrics"]

    metrics, extra = run_fn()
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"model": display_name, "setup": setup,
                   "metrics": metrics, **extra}, f, indent=2)
    return metrics


# ============================================================
# Table 1
# ============================================================

def create_table1(results):
    """results: list of (model display name, setup, metrics dict)."""

    rows = []

    for model_name, setup, metrics in results:

        rows.append(
            {
                "Model": model_name,
                "Setup": setup,
                **metrics,
            }
        )

    table = pd.DataFrame(rows)

    columns = [
        "Model",
        "Setup",
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
    ]

    table = table[columns]

    metric_columns = columns[2:]

    table[metric_columns] = (
        table[metric_columns].round(4)
    )

    return table


def table_to_latex(table):
    """LaTeX rows in the same format as Table 1 (best value per column in bold)."""
    metric_columns = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
    best = {c: table[c].max() for c in metric_columns}

    lines = [
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Model & Setup & Accuracy & Macro Prec. & Macro Rec. & Macro F1 \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        cells = [
            rf"\textbf{{{row[c]:.2f}}}" if row[c] == best[c] else f"{row[c]:.2f}"
            for c in metric_columns
        ]
        lines.append(
            rf"\textbf{{{row['Model']}}} & {row['Setup']} & " + " & ".join(cells) + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main():

    set_seed()

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Load DreamBank
    # --------------------------------------------------------

    df = load_dreams(DATA_PATH)

    # --------------------------------------------------------
    # 2. Sentence embeddings + global PCA
    # --------------------------------------------------------

    df = calculate_pca(df)

    # --------------------------------------------------------
    # 3. Four quadrant labels
    # --------------------------------------------------------

    df = assign_quadrants(df)

    # Save complete PCA data for reproducibility.
    df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "dreams_with_pca_and_quadrants.csv",
        ),
        index=False,
    )

    # --------------------------------------------------------
    # 4. Extreme cases
    # --------------------------------------------------------

    extreme_df = filter_extreme_cases(df)

    # --------------------------------------------------------
    # 5. 70 / 15 / 15
    # --------------------------------------------------------

    train_df, validation_df, test_df = (
        split_dataset(extreme_df)
    )

    train_df.to_csv(
        os.path.join(OUTPUT_DIR, "train.csv"),
        index=False,
    )

    validation_df.to_csv(
        os.path.join(OUTPUT_DIR, "validation.csv"),
        index=False,
    )

    test_df.to_csv(
        os.path.join(OUTPUT_DIR, "test.csv"),
        index=False,
    )

    # --------------------------------------------------------
    # 6. BERT / RoBERTa
    # --------------------------------------------------------

    results = []

    for display_name, model_name in MODELS.items():

        metrics = cached_run(
            display_name,
            "Supervised Fine-Tuning",
            lambda: train_model(
                display_name,
                model_name,
                train_df,
                validation_df,
                test_df,
            ),
        )

        results.append((display_name, "Supervised Fine-Tuning", metrics))

    # --------------------------------------------------------
    # 7. GPT-4o: Zero-Shot and Zero-Shot + CoT (same test set)
    # --------------------------------------------------------

    if os.environ.get("OPENAI_API_KEY"):

        for mode, setup in LLM_SETUPS.items():

            display_name = f"LLM ({LLM_MODEL}) {mode}"

            metrics = cached_run(
                display_name,
                setup,
                lambda: run_llm(mode, display_name, test_df),
            )

            results.append((f"LLM ({LLM_MODEL})", setup, metrics))

        if LLM_LIMIT:
            print(f"\nWARNING: LLM rows use only the first {LLM_LIMIT} test dreams "
                  f"(LLM_LIMIT) - not comparable to the full test set.")
    else:
        print("\nOPENAI_API_KEY not set - GPT-4o rows skipped.")

    # --------------------------------------------------------
    # 8. Generate Table 1
    # --------------------------------------------------------

    table1 = create_table1(results)

    print("\n")
    print("=" * 70)
    print(f"TABLE 1 (this run, extreme-cases test set, n={len(test_df)})")
    print("=" * 70)

    print(
        table1.to_string(
            index=False
        )
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        "table1_results.csv",
    )

    table1.to_csv(
        output_file,
        index=False,
    )

    latex_file = os.path.join(OUTPUT_DIR, "table1.tex")
    with open(latex_file, "w", encoding="utf-8") as f:
        f.write(table_to_latex(table1) + "\n")

    print(
        f"\nResults saved to: {output_file} and {latex_file}"
    )


if __name__ == "__main__":
    main()