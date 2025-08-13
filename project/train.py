import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from joblib import dump

from .utils import normalize_text


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load a CSV containing `text` and `label` columns and normalize text.

    Args:
        csv_path (Path): Path to a CSV file.

    Returns:
        pd.DataFrame: DataFrame with normalized text and labels.
    """
    df = pd.read_csv(csv_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have columns: text,label")
    df["text"] = df["text"].astype(str).map(normalize_text)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a baseline text classifier.")
    parser.add_argument("--task", default="sentiment", help="Task name (unused, for extensibility).")
    parser.add_argument("--lang", default="twi", help="Language code (unused, for extensibility).")
    parser.add_argument("--data_dir", default="data/public_dev", help="Directory containing train.csv and dev.csv.")
    parser.add_argument("--out_dir", default="models", help="Directory to save the trained model.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_csv = data_dir / "train.csv"
    dev_csv = data_dir / "dev.csv"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_data(train_csv)
    dev_df = load_data(dev_csv)

    # Build a simple TF‑IDF + Logistic Regression pipeline
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("lr", LogisticRegression(max_iter=1000)),
    ])

    pipe.fit(train_df["text"], train_df["label"])
    pred = pipe.predict(dev_df["text"])
    f1 = f1_score(dev_df["label"], pred, average="macro")
    print(f"Dev macro-F1: {f1:.3f}")

    # Save the model
    dump(pipe, out_dir / "baseline.joblib")


if __name__ == "__main__":
    main()