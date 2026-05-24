"""Preprocessing script: load raw dataset, clean, and write to data/processed/."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import DataLoader

RAW_PATH = ROOT / "data" / "MachineLearningRating_v3.txt"
OUT_PATH = ROOT / "data" / "processed" / "insurance_cleaned.csv"


def main() -> None:
    """Run the prepare stage: load raw data, drop zero-premium rows, persist cleaned CSV."""
    print(f"Loading raw data from {RAW_PATH}")
    loader = DataLoader(RAW_PATH)
    df = loader.load(sep="|")

    # Basic cleaning: drop rows where TotalPremium is zero or missing
    n_before = len(df)
    df = df[df["TotalPremium"] > 0].reset_index(drop=True)
    print(f"Dropped {n_before - len(df)} zero-premium rows. Remaining: {len(df)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Cleaned dataset written to {OUT_PATH}  ({df.shape[0]} rows x {df.shape[1]} cols)")


if __name__ == "__main__":
    main()
