import pytest
import pandas as pd
import numpy as np
from src.data_loader import DataLoader

SAMPLE_CSV = (
    "UnderwrittenCoverID|PolicyID|TransactionMonth|TotalPremium|TotalClaims"
    "|Province|Gender|VehicleType|Make|RegistrationYear\n"
    "1|P001|2014-02-01|500.0|0.0|Gauteng|Male|Passenger Vehicle|Toyota|2010\n"
    "2|P002|2014-03-01|800.0|1200.0|Western Cape|Female|Passenger Vehicle|Ford|2015\n"
    "3|P003|2014-04-01|600.0|0.0|KwaZulu-Natal|Male|Light Commercial|VW|2012\n"
)


@pytest.fixture
def loader(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text(SAMPLE_CSV)
    ld = DataLoader(p)
    ld.load(sep="|")
    return ld


def test_load_returns_dataframe(loader):
    assert isinstance(loader.df, pd.DataFrame)
    assert len(loader.df) == 3


def test_derived_columns_present(loader):
    for col in ("LossRatio", "Margin", "HasClaim"):
        assert col in loader.df.columns, f"{col} missing"


def test_loss_ratio_calculation(loader):
    row = loader.df[loader.df["PolicyID"] == "P002"].iloc[0]
    assert abs(row["LossRatio"] - 1200.0 / 800.0) < 1e-6


def test_margin_calculation(loader):
    row = loader.df[loader.df["PolicyID"] == "P001"].iloc[0]
    assert abs(row["Margin"] - 500.0) < 1e-6


def test_has_claim_flag(loader):
    assert loader.df[loader.df["PolicyID"] == "P001"]["HasClaim"].iloc[0] == 0
    assert loader.df[loader.df["PolicyID"] == "P002"]["HasClaim"].iloc[0] == 1


def test_transaction_month_is_datetime(loader):
    assert pd.api.types.is_datetime64_any_dtype(loader.df["TransactionMonth"])


def test_zero_premium_loss_ratio_is_nan(tmp_path):
    csv = (
        "PolicyID|TotalPremium|TotalClaims\n"
        "P999|0.0|100.0\n"
    )
    p = tmp_path / "zero.csv"
    p.write_text(csv)
    ld = DataLoader(p)
    df = ld.load(sep="|")
    assert np.isnan(df.loc[0, "LossRatio"])


def test_validate_schema_identifies_missing(loader):
    result = loader.validate_schema()
    # sample CSV is missing many expected columns
    assert len(result["missing_columns"]) > 0


def test_validate_schema_no_extra_on_full_schema(tmp_path):
    from src.data_loader import EXPECTED_COLUMNS
    numeric_cols = {
        "TotalPremium", "TotalClaims", "SumInsured", "CalculatedPremiumPerTerm",
        "CustomValueEstimate", "CapitalOutstanding", "Kilowatts", "Cubiccapacity",
        "NumberOfDoors", "NumberOfVehiclesInFleet", "RegistrationYear",
        "ExcessSelected", "Cylinders",
    }
    cols = list(EXPECTED_COLUMNS)
    row_vals = ["100.0" if c in numeric_cols else "val" for c in cols]
    p = tmp_path / "full.csv"
    p.write_text("|".join(cols) + "\n" + "|".join(row_vals) + "\n")
    ld = DataLoader(p)
    ld.load(sep="|")
    result = ld.validate_schema()
    derived = {"LossRatio", "Margin", "HasClaim", "VehicleAge"}
    assert result["extra_columns"] - derived == set()
    assert result["missing_columns"] == set()
