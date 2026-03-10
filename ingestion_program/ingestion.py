# ingestion_program/ingestion.py
import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd

import zipfile


def import_submission(submission_dir: Path):
    # 1) Cas simple : submission.py déjà extrait
    direct_candidates = [
        submission_dir / "submission.py",
        submission_dir / "solution" / "submission.py",
    ]
    for path in direct_candidates:
        if path.exists():
            submission_path = path
            spec = importlib.util.spec_from_file_location("submission", str(submission_path))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore
            return module

    # 2) Chercher un zip dans le dossier de soumission
    zip_files = list(submission_dir.glob("*.zip"))
    if len(zip_files) == 1:
        zip_path = zip_files[0]
        extract_dir = submission_dir / "_unzipped_submission"
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        extracted_candidates = [
            extract_dir / "submission.py",
            extract_dir / "solution" / "submission.py",
        ]

        # recherche récursive de secours
        recursive_matches = list(extract_dir.rglob("submission.py"))
        for path in extracted_candidates:
            if path.exists():
                submission_path = path
                break
        else:
            if len(recursive_matches) == 1:
                submission_path = recursive_matches[0]
            elif len(recursive_matches) > 1:
                raise FileNotFoundError(
                    "Multiple submission.py files found after unzip: "
                    + ", ".join(str(p) for p in recursive_matches)
                )
            else:
                extracted_files = [str(p) for p in extract_dir.rglob("*") if p.is_file()]
                raise FileNotFoundError(
                    "No submission.py found after unzipping submission. "
                    f"Files found: {extracted_files}"
                )

        spec = importlib.util.spec_from_file_location("submission", str(submission_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore
        return module

    # 3) Debug utile si rien n'est trouvé
    all_files = [str(p) for p in submission_dir.rglob("*") if p.is_file()]
    raise FileNotFoundError(
        "Could not find submission.py or a single zip file in submission directory. "
        f"Files found in {submission_dir}: {all_files}"
    )


def load_features(data_dir: Path, split: str) -> pd.DataFrame:
    split_dir = data_dir / split
    path = split_dir / f"{split}_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing features file: {path}")
    return pd.read_csv(path)


def load_labels(data_dir: Path, split: str) -> pd.Series:
    split_dir = data_dir / split
    path = split_dir / f"{split}_labels.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing labels file: {path}")

    y_df = pd.read_csv(path)
    if "label" in y_df.columns:
        return y_df["label"]
    return y_df.iloc[:, 0]


def save_predictions(preds, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"label": preds}).to_csv(out_path, index=False)


def assert_same_columns(X_train: pd.DataFrame, X_other: pd.DataFrame, name: str):
    if list(X_train.columns) != list(X_other.columns):
        if set(X_train.columns) == set(X_other.columns):
            X_other = X_other[X_train.columns]
            return X_other
        raise ValueError(
            f"Feature columns mismatch between train and {name}.\n"
            f"train cols={list(X_train.columns)}\n"
            f"{name} cols={list(X_other.columns)}"
        )
    return X_other


def get_model_from_submission(subm):
    if hasattr(subm, "get_model"):
        model = subm.get_model()
        return model
    if hasattr(subm, "Model"):
        return subm.Model()
    raise AttributeError("submission.py must define get_model() or a Model class")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--submission-dir", type=str, required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    submission_dir = Path(args.submission_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Ingestion started")
    print("Data dir:", data_dir.resolve())
    print("Submission dir:", submission_dir.resolve())
    print("Output dir:", output_dir.resolve())

    t0 = time.time()

    X_train = load_features(data_dir, "train")
    y_train = load_labels(data_dir, "train")

    X_test = load_features(data_dir, "test")
    X_private = load_features(data_dir, "private_test")

    X_test = assert_same_columns(X_train, X_test, "test")
    X_private = assert_same_columns(X_train, X_private, "private_test")

    print("Shapes:")
    print("  train:", X_train.shape, "labels:", y_train.shape)
    print("  test:", X_test.shape)
    print("  private_test:", X_private.shape)

    # Load submission and model
    subm = import_submission(submission_dir)
    model = get_model_from_submission(subm)

    # Fit
    print("Fitting model...")
    model.fit(X_train, y_train)

    # Predict
    print("Predicting test...")
    pred_test = model.predict(X_test)
    print("Predicting private_test...")
    pred_private = model.predict(X_private)

    pred_test = np.asarray(pred_test)
    pred_private = np.asarray(pred_private)

    if pred_test.shape[0] != X_test.shape[0]:
        raise ValueError("Wrong number of predictions for test set.")
    if pred_private.shape[0] != X_private.shape[0]:
        raise ValueError("Wrong number of predictions for private_test set.")

    # Save
    save_predictions(pred_test, output_dir / "test_predictions.csv")
    save_predictions(pred_private, output_dir / "private_test_predictions.csv")

    elapsed = time.time() - t0
    print(f"Done. Runtime: {elapsed:.2f}s")
    print("Wrote files:")
    print("  test_predictions.csv")
    print("  private_test_predictions.csv")


if __name__ == "__main__":
    main()