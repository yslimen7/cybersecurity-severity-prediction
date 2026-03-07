import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

EVAL_SETS = ["test", "private_test"]


def compute_accuracy(predictions, targets):
    # Work explicitly on the "label" column and avoid NaN issues
    predictions = predictions["label"].fillna(-10)
    targets = targets["label"].fillna(-10)
    score = f1_score(targets, predictions, average="macro")
    return score


def validate_columns(df, file_path):
    required_columns = {"label"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"File '{file_path}' is missing required column(s): "
            f"{sorted(missing)}. Found columns: {list(df.columns)}"
        )


def main(reference_dir, prediction_dir, output_dir):
    scores = {}
    for eval_set in EVAL_SETS:
        print(f"Scoring {eval_set}")

        prediction_file = prediction_dir / f"{eval_set}_predictions.csv"
        target_file = reference_dir / f"{eval_set}_labels.csv"

        if not prediction_file.exists():
            raise FileNotFoundError(
                f"Missing prediction file: '{prediction_file}'"
            )
        if not target_file.exists():
            raise FileNotFoundError(
                f"Missing reference file: '{target_file}'"
            )

        predictions = pd.read_csv(prediction_file)
        targets = pd.read_csv(target_file)

        validate_columns(predictions, prediction_file)
        validate_columns(targets, target_file)

        if len(predictions) != len(targets):
            raise ValueError(
                f"Row count mismatch for '{eval_set}': "
                f"{len(predictions)} prediction rows vs "
                f"{len(targets)} target rows"
            )

        scores[eval_set] = float(compute_accuracy(predictions, targets))

    # Add duration metadata only when available
    metadata_path = prediction_dir / "metadata.json"
    if metadata_path.exists():
        json_durations = metadata_path.read_text()
        durations = json.loads(json_durations)
        if isinstance(durations, dict):
            scores.update(**durations)
    else:
        print(
            f"Warning: {metadata_path} not found. "
            "Skipping duration metadata."
        )

    print(scores)

    # Write output scores
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scores.json").write_text(json.dumps(scores))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scoring program for codabench"
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        default="/app/input/ref",
        help="",
    )
    parser.add_argument(
        "--prediction-dir",
        type=str,
        default="/app/input/res",
        help="",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/app/output",
        help="",
    )

    args = parser.parse_args()

    main(
        Path(args.reference_dir),
        Path(args.prediction_dir),
        Path(args.output_dir)
    )