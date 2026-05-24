import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Schema constants — centralised so tests and validation scripts share them
# ---------------------------------------------------------------------------

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

DATETIME_COLS = ("TransactionMonth", "VehicleIntroDate")

# Columns that must be positive for derived metrics to be meaningful
CORE_NUMERIC_COLS = {"TotalPremium", "TotalClaims"}
class DataLoader:
    """Load, validate, and enrich the ACIS auto-insurance dataset.

    Typical usage::

        loader = DataLoader("data/MachineLearningRating_v3.txt")
        df = loader.load(sep="|")
        report = loader.validate_schema()
    """

    def __init__(self, data_path: str | Path) -> None:
        """Initialise the loader for *data_path*.

        Args:
            data_path: Path to the pipe-delimited raw dataset.

        Raises:
            TypeError: If *data_path* is not a str or Path-like object.
        """
        if not isinstance(data_path, (str, Path)):
            raise TypeError(
                f"data_path must be str or Path, got {type(data_path).__name__}"
            )
        self.data_path = Path(data_path)
        self.df: pd.DataFrame | None = None

    def load(self, sep: str = "|", **kwargs) -> pd.DataFrame:
        """Read the raw file, coerce dtypes, and add derived columns.

        Args:
            sep: Column delimiter (default ``"|"``).
            **kwargs: Extra keyword arguments forwarded to :func:`pandas.read_csv`.

        Returns:
            The enriched :class:`~pandas.DataFrame` (also stored as ``self.df``).

        Raises:
            FileNotFoundError: If *data_path* does not exist.
            ValueError: If the file is empty or cannot be parsed as a table.
            OSError: On any other I/O failure.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        try:
            self.df = pd.read_csv(
                self.data_path, sep=sep, low_memory=False, **kwargs
            )
        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"Data file is empty: {self.data_path}") from exc
        except pd.errors.ParserError as exc:
            raise ValueError(
                f"Could not parse '{self.data_path}' with sep='{sep}': {exc}"
            ) from exc
        except OSError as exc:
            raise OSError(f"I/O error reading '{self.data_path}': {exc}") from exc

        if self.df.empty:
            raise ValueError(f"Loaded DataFrame is empty: {self.data_path}")

        self._fix_dtypes()
        self._add_derived_metrics()
        return self.df

    def validate_schema(self) -> dict:
        """Return missing and unexpected column names compared to EXPECTED_COLUMNS.

        Returns:
            Dictionary with keys ``"missing_columns"`` and ``"extra_columns"``,
            each holding a :class:`set` of column name strings.

        Raises:
            RuntimeError: If :meth:`load` has not been called yet.
        """
        if self.df is None:
            raise RuntimeError("Call load() before validate_schema().")
        present = set(self.df.columns)
        return {
            "missing_columns": EXPECTED_COLUMNS - present,
            "extra_columns": present - EXPECTED_COLUMNS,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fix_dtypes(self) -> None:
        """Coerce datetime and categorical columns to appropriate dtypes in-place."""
        for col in DATETIME_COLS:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors="coerce")

        for col in CATEGORICAL_COLS:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype("category")

        if "PostalCode" in self.df.columns:
            self.df["PostalCode"] = self.df["PostalCode"].astype(str)

    def _add_derived_metrics(self) -> None:
        """Add LossRatio, Margin, HasClaim, and VehicleAge columns in-place."""
        if CORE_NUMERIC_COLS.issubset(self.df.columns):
            # Guard against division by zero — zero-premium rows yield NaN
            safe_premium = self.df["TotalPremium"].replace(0, np.nan)
            self.df["LossRatio"] = self.df["TotalClaims"] / safe_premium
            self.df["Margin"] = self.df["TotalPremium"] - self.df["TotalClaims"]
            self.df["HasClaim"] = (self.df["TotalClaims"] > 0).astype(int)

        if "RegistrationYear" in self.df.columns and "TransactionMonth" in self.df.columns:
            self.df["VehicleAge"] = (
                self.df["TransactionMonth"].dt.year - self.df["RegistrationYear"]
            ).clip(lower=0)
