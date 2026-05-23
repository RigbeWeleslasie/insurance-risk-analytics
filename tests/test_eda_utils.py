import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # must precede pyplot import — non-interactive backend for CI
import matplotlib.pyplot as plt  # noqa: E402
from src import eda_utils  # noqa: E402


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "PolicyID": [f"P{i:04d}" for i in range(n)],
        "TotalPremium": rng.uniform(200, 2000, n),
        "TotalClaims": np.where(rng.random(n) < 0.25, rng.uniform(100, 5000, n), 0.0),
        "Province": pd.Categorical(rng.choice(["Gauteng", "Western Cape", "KwaZulu-Natal"], n)),
        "VehicleType": pd.Categorical(rng.choice(["Passenger", "Commercial"], n)),
        "Gender": pd.Categorical(rng.choice(["Male", "Female"], n)),
        "HasClaim": rng.integers(0, 2, n),
    })
    df["LossRatio"] = df["TotalClaims"] / df["TotalPremium"]
    df["Margin"] = df["TotalPremium"] - df["TotalClaims"]
    return df


def test_missing_value_report_empty_when_no_nulls(sample_df):
    report = eda_utils.missing_value_report(sample_df)
    assert isinstance(report, pd.DataFrame)
    assert len(report) == 0


def test_missing_value_report_detects_nulls(sample_df):
    sample_df = sample_df.copy()
    sample_df.loc[0, "TotalPremium"] = np.nan
    report = eda_utils.missing_value_report(sample_df)
    assert "TotalPremium" in report.index


def test_loss_ratio_by_group_has_expected_columns(sample_df):
    result = eda_utils.loss_ratio_by_group(sample_df, "Province")
    for col in ("TotalPremium", "TotalClaims", "LossRatio", "Margin"):
        assert col in result.columns


def test_loss_ratio_by_group_sorted_descending(sample_df):
    result = eda_utils.loss_ratio_by_group(sample_df, "Province")
    ratios = result["LossRatio"].tolist()
    assert ratios == sorted(ratios, reverse=True)


def test_outlier_summary_returns_correct_columns(sample_df):
    result = eda_utils.outlier_summary(sample_df, ["TotalPremium", "TotalClaims"])
    for col in ("Q1", "Q3", "IQR", "n_outliers", "pct_outliers"):
        assert col in result.columns


def test_plot_numerical_distributions_returns_figure(sample_df):
    fig = eda_utils.plot_numerical_distributions(sample_df, ["TotalPremium", "TotalClaims", "Margin"])
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_plot_boxplots_returns_figure(sample_df):
    fig = eda_utils.plot_boxplots(sample_df, ["TotalPremium", "TotalClaims"])
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_plot_correlation_matrix_returns_figure(sample_df):
    fig = eda_utils.plot_correlation_matrix(sample_df, ["TotalPremium", "TotalClaims", "LossRatio"])
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_claim_stats_by_group(sample_df):
    result = eda_utils.claim_stats_by_group(sample_df, "Province")
    assert "ClaimFrequency" in result.columns
    assert all((result["ClaimFrequency"] >= 0) & (result["ClaimFrequency"] <= 1))
