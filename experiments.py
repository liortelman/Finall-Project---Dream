"""
Reproduce the supervised-model rows of Table 1 in the final report.

Pipeline:
    Dream texts
        -> SentenceTransformer embeddings
        -> PCA (2 dimensions)
        -> 4 quadrant labels
        -> extreme-case filtering
        -> 70/15/15 train/validation/test split
        -> BERT / RoBERTa fine-tuning
        -> Accuracy / Macro Precision / Macro Recall / Macro F1

NOTE:
The GPT-4o rows from Table 1 require API inference and are intentionally
not fabricated here. They can be added using the same fixed test set.
"""

import os
import random
import numpy as np
import pandas as pd
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

DATA_PATH = "/Users/morberger/PycharmProjects/Finall-Project---Dream/PCA/all_dreams_combined.csv"

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

# "Extreme cases" are points sufficiently far from the PCA axes.
#
# The report says that extreme cases were used, but does not provide
# an exact numerical threshold. We therefore expose the percentile
# explicitly instead of hiding an arbitrary choice.
#
# Keep the most extreme 50% by |PCA1| and |PCA2| independently.
EXTREME_PERCENTILE = 50

OUTPUT_DIR = "table1_outputs"


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

    df[TEXT_COLUMN] = (
        df[TEXT_COLUMN]
        .astype(str)
        .str.strip()
    )

    df = df[df[TEXT_COLUMN].str.len() > 0]

    print(f"Loaded {len(df):,} dreams.")

    return df.reset_index(drop=True)


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

    return result


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

    The written report describes percentile-based extreme-case
    filtering but does not specify its exact numerical percentile.
    For reproducibility we make the threshold explicit.
    """

    pca1_threshold = np.percentile(
        np.abs(df["pca1"]),
        EXTREME_PERCENTILE,
    )

    pca2_threshold = np.percentile(
        np.abs(df["pca2"]),
        EXTREME_PERCENTILE,
    )

    filtered = df[
        (np.abs(df["pca1"]) >= pca1_threshold)
        & (np.abs(df["pca2"]) >= pca2_threshold)
    ].copy()

    print("\nExtreme-case thresholds:")
    print(f"|PCA1| >= {pca1_threshold:.4f}")
    print(f"|PCA2| >= {pca2_threshold:.4f}")

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
            max_length=512,
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

    training_args = TrainingArguments(
        output_dir=model_output_dir,

        learning_rate=LEARNING_RATE,

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,

        num_train_epochs=NUM_EPOCHS,

        weight_decay=0.01,

        eval_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        logging_strategy="epoch",

        report_to="none",

        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
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
        [TEXT_COLUMN, "pca1", "pca2", "label", "quadrant"]
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

    return metrics


# ============================================================
# Table 1
# ============================================================

def create_table1(results):

    rows = []

    for model_name, metrics in results.items():

        rows.append(
            {
                "Model": model_name,
                "Setup": "Supervised Fine-Tuning",
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

    results = {}

    for display_name, model_name in MODELS.items():

        metrics = train_model(
            display_name,
            model_name,
            train_df,
            validation_df,
            test_df,
        )

        results[display_name] = metrics

    # --------------------------------------------------------
    # 7. Generate Table 1 results
    # --------------------------------------------------------

    table1 = create_table1(results)

    print("\n")
    print("=" * 70)
    print("TABLE 1 — REPRODUCED SUPERVISED RESULTS")
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

    print(
        f"\nResults saved to: {output_file}"
    )


if __name__ == "__main__":
    main()