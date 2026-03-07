# Script to download the data from a given source and create the splits
# This is a mock version that generate fake problems

#OUR COMMENTS ----------------------------------------------
"""
This dataset does not contain a predefined Severity Level.
We construct an artificial but realistic severity level based on
the operational and financial impact of each cyber incident.

Severity is derived from:
- Financial Loss (economic impact)
- Number of Affected Users (social impact)
- Incident Resolution Time (operational complexity)

We compute an impact score by normalizing these variables and
splitting the score into 4 quantiles:
Low / Medium / High / Critical

IMPORTANT: 
To reduce direct target leakage, we REMOVE "Financial Loss"
from the feature set after constructing Severity.

We keep:
- Number of Affected Users
- Incident Resolution Time

This simulates a realistic severity prediction scenario.
"""
#-----------------------------------------------------------

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

PHASE = 'dev_phase'

DATA_DIR = Path(PHASE) / 'input_data'
REF_DIR = Path(PHASE) / 'reference_data'

DATA_PATH = Path("data/global_cyber_threats.csv")

YEAR_COL = "Year"
LOSS_COL = "Financial Loss (in Million $)"
USERS_COL = "Number of Affected Users"
TIME_COL = "Incident Resolution Time (in Hours)"
TARGET_COL = "Severity"

def make_csv(data, filepath):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(filepath, index=False)

def normalize_column(col):
    return (col - col.min()) / (col.max() - col.min())

if __name__ == "__main__":

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    df.columns = [c.strip() for c in df.columns]

    # Basic cleaning
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df = df.dropna(subset=[YEAR_COL])

    # Fill NA
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("Unknown")

    for col in df.select_dtypes(include=["number"]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # ---------------------------------------
    # CREATE SEVERITY LEVEL
    # ---------------------------------------

    print("Creating Severity label...")

    df["loss_norm"] = normalize_column(df[LOSS_COL])
    df["users_norm"] = normalize_column(df[USERS_COL])
    df["time_norm"] = normalize_column(df[TIME_COL])

    df["impact_score"] = (
        df["loss_norm"] +
        df["users_norm"] +
        df["time_norm"]
    )
    df[TARGET_COL] = pd.qcut(
        df["impact_score"],
        q=4,
        labels=["Low", "Medium", "High", "Critical"]
    )

    # ---------------------------------------
    # FEATURE SELECTION DECISION
    # ---------------------------------------
    # Remove Financial Loss to reduce direct leakage
    df = df.drop(columns=[LOSS_COL, "loss_norm", "users_norm", "time_norm", "impact_score"])

    # ---------------------------------------
    # TEMPORAL SPLIT
    # ---------------------------------------

    train_df = df[df[YEAR_COL].between(2015, 2022)]
    test_df = df[df[YEAR_COL] == 2023]
    private_test_df = df[df[YEAR_COL] == 2024]

    if train_df.empty or test_df.empty or private_test_df.empty:
        raise ValueError("One of the splits is empty. Check available years in dataset.")

    # Separate X / y
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]

    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    X_private = private_test_df.drop(columns=[TARGET_COL])
    y_private = private_test_df[TARGET_COL]

    # ---------------------------------------
    # SAVE IN TEMPLATE FORMAT
    # ---------------------------------------

    for split, X_split, y_split in [
        ("train", X_train, y_train),
        ("test", X_test, y_test),
        ("private_test", X_private, y_private),
    ]:
        split_dir = DATA_DIR / split
        make_csv(X_split, split_dir / f"{split}_features.csv")

        label_dir = split_dir if split == "train" else REF_DIR
        make_csv({"label": y_split}, label_dir / f"{split}_labels.csv")

    print("OK; dev_phase successfully generated.")
    print("Train shape:", X_train.shape)
    print("Public test shape:", X_test.shape)
    print("Private test shape:", X_private.shape)
