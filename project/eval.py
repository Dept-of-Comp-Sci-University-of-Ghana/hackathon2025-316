import argparse
import json
from pathlib import Path

import pandas as pd
from joblib import load
from sklearn.metrics import f1_score

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
    parser = argparse.ArgumentParser(description="Evaluate a trained text classifier.")
    parser.add_argument("--task", default="sentiment", help="Task name (unused, for extensibility).")
    parser.add_argument("--lang", default="twi", help="Language code (unused, for extensibility).")
    parser.add_argument("--data_dir", default="data/public_dev", help="Directory containing dev.csv.")
    parser.add_argument("--model_dir", default="models", help="Directory containing the saved model.")
    parser.add_argument("--out", default="reports/metrics.json", help="Path to write metrics JSON.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    dev_csv = data_dir / "dev.csv"
    model_dir = Path(args.model_dir)
    out_path = Path(args.out)

    # Load model and data
    model = load(model_dir / "baseline.joblib")
    dev_df = load_data(dev_csv)

    # Make predictions and compute macro-F1
    pred = model.predict(dev_df["text"])
    f1 = f1_score(dev_df["label"], pred, average="macro")

    # Write metrics
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {"task": args.task, "lang": args.lang, "f1_macro": float(f1)}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()