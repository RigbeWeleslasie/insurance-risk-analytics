"""Validation script: sanity-check the processed dataset before modelling."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed" / "insurance_cleaned.csv"


def main() -> None:
    """Assert minimum quality constraints on the cleaned dataset."""
    df = pd.read_csv(PROCESSED)
    assert len(df) > 0, "Processed dataset is empty"
    for col in ("TotalPremium", "TotalClaims", "LossRatio", "Margin", "HasClaim"):
        assert col in df.columns, f"Missing required column: {col}"
    assert (df["TotalPremium"] > 0).all(), "Zero-premium rows found in cleaned data"
    print(f"Validation passed: {len(df):,} rows, {df.shape[1]} columns")


if __name__ == "__main__":
    main()
