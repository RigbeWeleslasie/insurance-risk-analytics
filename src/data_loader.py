import pandas as pd
import numpy as np
from pathlib import Path

EXPECTED_COLUMNS = {
    "UnderwrittenCoverID", "PolicyID", "TransactionMonth",
    "IsVATRegistered", "Citizenship", "LegalType", "Title", "Language",
    "Bank", "AccountType", "MaritalStatus", "Gender",
    "Country", "Province", "PostalCode", "MainCrestaZone", "SubCrestaZone",
    "ItemType", "Mmcode", "VehicleType", "RegistrationYear", "Make", "Model",
    "Cylinders", "Cubiccapacity", "Kilowatts", "Bodytype", "NumberOfDoors",
    "VehicleIntroDate", "CustomValueEstimate", "AlarmImmobiliser",
    "TrackingDevice", "CapitalOutstanding", "NewVehicle", "WrittenOff",
    "Rebuilt", "Converted", "CrossBorder", "NumberOfVehiclesInFleet",
    "SumInsured", "TermFrequency", "CalculatedPremiumPerTerm", "ExcessSelected",
    "CoverCategory", "CoverType", "CoverGroup", "Section", "Product",
    "StatutoryClass", "StatutoryRiskType",
    "TotalPremium", "TotalClaims",
}

CATEGORICAL_COLS = [
    "Province", "Gender", "VehicleType", "Make", "Model",
    "CoverType", "CoverCategory", "CoverGroup", "MaritalStatus",
    "LegalType", "Bank", "AccountType", "Bodytype", "Section",
    "Product", "StatutoryClass", "StatutoryRiskType", "Citizenship",
    "AlarmImmobiliser", "TrackingDevice", "NewVehicle", "WrittenOff",
    "Rebuilt", "Converted", "CrossBorder",
]


class DataLoader:
    """Loads, validates, and enriches the ACIS insurance dataset."""

    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)
        self.df: pd.DataFrame | None = None

    def load(self, sep: str = "|", **kwargs) -> pd.DataFrame:
        """Read the CSV, fix dtypes, and add derived columns."""
        self.df = pd.read_csv(self.data_path, sep=sep, low_memory=False, **kwargs)
        self._fix_dtypes()
        self._add_derived_metrics()
        return self.df

    def validate_schema(self) -> dict:
        """Return sets of missing and unexpected column names."""
        if self.df is None:
            raise RuntimeError("Call load() before validate_schema().")
        present = set(self.df.columns)
        return {
            "missing_columns": EXPECTED_COLUMNS - present,
            "extra_columns": present - EXPECTED_COLUMNS,
        }

    def _fix_dtypes(self) -> None:
        for col in ("TransactionMonth", "VehicleIntroDate"):
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors="coerce")

        for col in CATEGORICAL_COLS:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype("category")

        if "PostalCode" in self.df.columns:
            self.df["PostalCode"] = self.df["PostalCode"].astype(str)

    def _add_derived_metrics(self) -> None:
        if {"TotalPremium", "TotalClaims"}.issubset(self.df.columns):
            safe_premium = self.df["TotalPremium"].replace(0, np.nan)
            self.df["LossRatio"] = self.df["TotalClaims"] / safe_premium
            self.df["Margin"] = self.df["TotalPremium"] - self.df["TotalClaims"]
            self.df["HasClaim"] = (self.df["TotalClaims"] > 0).astype(int)

        if "RegistrationYear" in self.df.columns and "TransactionMonth" in self.df.columns:
            self.df["VehicleAge"] = (
                self.df["TransactionMonth"].dt.year - self.df["RegistrationYear"]
            ).clip(lower=0)
