"""Statistical hypothesis testing utilities (Task 3)."""
"""
src/hypothesis_tests.py
-----------------------
Reusable statistical test functions for ACIS A/B hypothesis testing (Task 3).

Tests covered
-------------
1. risk_diff_provinces        – H1: No risk difference across provinces
2. risk_diff_zipcodes         – H2: No risk difference across zip codes
3. margin_diff_zipcodes       – H3: No margin difference across zip codes
4. risk_diff_gender           – H4: No risk difference between Women and Men

Each function returns a dict with keys:
    hypothesis, kpi, test_used, statistic, p_value, decision, interpretation
"""

import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _welch_ttest(group_a: pd.Series, group_b: pd.Series, alpha: float = 0.05) -> dict:
    """Two-sided Welch's t-test. Returns statistic, p_value, decision."""
    stat, p = stats.ttest_ind(group_a.dropna(), group_b.dropna(), equal_var=False)
    return {"statistic": round(float(stat), 4),
            "p_value": round(float(p), 6),
            "decision": "Reject H₀" if p < alpha else "Fail to Reject H₀"}


def _ztest(group_a: pd.Series, group_b: pd.Series, alpha: float = 0.05) -> dict:
    """
    Two-proportion z-test for claim frequency (proportions).
    group_a / group_b should be boolean Series (1 = claim, 0 = no claim).
    """
    from statsmodels.stats.proportion import proportions_ztest
    count = np.array([group_a.sum(), group_b.sum()])
    nobs  = np.array([len(group_a), len(group_b)])
    stat, p = proportions_ztest(count, nobs)
    return {"statistic": round(float(stat), 4),
            "p_value": round(float(p), 6),
            "decision": "Reject H₀" if p < alpha else "Fail to Reject H₀"}


def _chi2_test(df: pd.DataFrame, col_a: str, col_b: str, alpha: float = 0.05) -> dict:
    """Chi-squared test of independence between two categorical columns."""
    ct = pd.crosstab(df[col_a], df[col_b])
    stat, p, dof, _ = stats.chi2_contingency(ct)
    return {"statistic": round(float(stat), 4),
            "p_value": round(float(p), 6),
            "dof": dof,
            "decision": "Reject H₀" if p < alpha else "Fail to Reject H₀"}


def _ensure_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Derive HasClaim, LossRatio, Margin if not already present."""
    df = df.copy()
    if "HasClaim" not in df.columns:
        df["HasClaim"] = (df["TotalClaims"] > 0).astype(int)
    if "LossRatio" not in df.columns:
        df["LossRatio"] = df["TotalClaims"] / df["TotalPremium"].replace(0, np.nan)
    if "Margin" not in df.columns:
        df["Margin"] = df["TotalPremium"] - df["TotalClaims"]
    return df


# ---------------------------------------------------------------------------
# H1 — Risk differences across provinces
# ---------------------------------------------------------------------------

def risk_diff_provinces(
    df: pd.DataFrame,
    kpi: str = "ClaimSeverity",
    province_a: str = "Gauteng",
    province_b: str = "Western Cape",
    alpha: float = 0.05,
) -> dict:
    """
    H₀: There are no risk differences across provinces.

    Parameters
    ----------
    kpi : 'ClaimFrequency' | 'ClaimSeverity'
    province_a / province_b : the two provinces to compare
    """
    df = _ensure_metrics(df)

    if kpi == "ClaimFrequency":
        a = df.loc[df["Province"] == province_a, "HasClaim"]
        b = df.loc[df["Province"] == province_b, "HasClaim"]
        result = _ztest(a, b, alpha)
        test_name = "Two-proportion z-test"
        kpi_label = "Claim Frequency"
    else:  # ClaimSeverity
        a = df.loc[(df["Province"] == province_a) & (df["HasClaim"] == 1), "TotalClaims"]
        b = df.loc[(df["Province"] == province_b) & (df["HasClaim"] == 1), "TotalClaims"]
        result = _welch_ttest(a, b, alpha)
        test_name = "Welch's t-test"
        kpi_label = "Claim Severity (ZAR)"

    mean_a = a.mean()
    mean_b = b.mean()
    pct_diff = ((mean_b - mean_a) / mean_a * 100) if mean_a != 0 else np.nan

    if result["decision"] == "Reject H₀":
        interpretation = (
            f"We REJECT H₀ for provinces (p = {result['p_value']:.4f} < {alpha}). "
            f"{province_a} shows a mean {kpi_label} of {mean_a:,.2f} vs "
            f"{province_b}'s {mean_b:,.2f} — a {abs(pct_diff):.1f}% difference. "
            f"Province-level rating factors are statistically justified."
        )
    else:
        interpretation = (
            f"We FAIL TO REJECT H₀ for provinces (p = {result['p_value']:.4f} ≥ {alpha}). "
            f"No statistically significant {kpi_label} difference between "
            f"{province_a} and {province_b} at this sample."
        )

    return {
        "hypothesis": "H₁: No risk difference across provinces",
        "group_a": province_a,
        "group_b": province_b,
        "kpi": kpi_label,
        "test_used": test_name,
        "mean_a": round(mean_a, 4),
        "mean_b": round(mean_b, 4),
        **result,
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# H2 — Risk differences across zip codes
# ---------------------------------------------------------------------------

def risk_diff_zipcodes(
    df: pd.DataFrame,
    kpi: str = "ClaimSeverity",
    top_n: int = 5,
    alpha: float = 0.05,
) -> list[dict]:
    """
    H₀: There are no risk differences between zip codes.

    Selects the top_n postal codes by policy count and runs pairwise tests.
    Returns a list of result dicts (one per pair).
    """
    df = _ensure_metrics(df)
    top_codes = (
        df["PostalCode"].value_counts().head(top_n).index.tolist()
    )

    results = []
    for code_a, code_b in combinations(top_codes, 2):
        if kpi == "ClaimFrequency":
            a = df.loc[df["PostalCode"] == code_a, "HasClaim"]
            b = df.loc[df["PostalCode"] == code_b, "HasClaim"]
            result = _ztest(a, b, alpha)
            test_name = "Two-proportion z-test"
            kpi_label = "Claim Frequency"
        else:
            a = df.loc[(df["PostalCode"] == code_a) & (df["HasClaim"] == 1), "TotalClaims"]
            b = df.loc[(df["PostalCode"] == code_b) & (df["HasClaim"] == 1), "TotalClaims"]
            if len(a) < 5 or len(b) < 5:
                continue
            result = _welch_ttest(a, b, alpha)
            test_name = "Welch's t-test"
            kpi_label = "Claim Severity (ZAR)"

        mean_a = a.mean()
        mean_b = b.mean()

        if result["decision"] == "Reject H₀":
            interpretation = (
                f"Zip {code_a} vs {code_b}: REJECT H₀ (p = {result['p_value']:.4f}). "
                f"Mean {kpi_label}: {mean_a:,.2f} vs {mean_b:,.2f}. "
                f"Postal-code risk loading is warranted."
            )
        else:
            interpretation = (
                f"Zip {code_a} vs {code_b}: FAIL TO REJECT H₀ (p = {result['p_value']:.4f}). "
                f"No significant {kpi_label} difference detected."
            )

        results.append({
            "hypothesis": "H₂: No risk difference across zip codes",
            "group_a": str(code_a),
            "group_b": str(code_b),
            "kpi": kpi_label,
            "test_used": test_name,
            "mean_a": round(mean_a, 4),
            "mean_b": round(mean_b, 4),
            **result,
            "interpretation": interpretation,
        })
    return results


# ---------------------------------------------------------------------------
# H3 — Margin differences across zip codes
# ---------------------------------------------------------------------------

def margin_diff_zipcodes(
    df: pd.DataFrame,
    top_n: int = 5,
    alpha: float = 0.05,
) -> list[dict]:
    """
    H₀: There is no significant margin (profit) difference between zip codes.

    Uses Welch's t-test on Margin = TotalPremium − TotalClaims.
    """
    df = _ensure_metrics(df)
    top_codes = df["PostalCode"].value_counts().head(top_n).index.tolist()

    results = []
    for code_a, code_b in combinations(top_codes, 2):
        a = df.loc[df["PostalCode"] == code_a, "Margin"]
        b = df.loc[df["PostalCode"] == code_b, "Margin"]
        result = _welch_ttest(a, b, alpha)
        mean_a = a.mean()
        mean_b = b.mean()

        if result["decision"] == "Reject H₀":
            interpretation = (
                f"Zip {code_a} vs {code_b}: REJECT H₀ (p = {result['p_value']:.4f}). "
                f"Mean Margin: ZAR {mean_a:,.2f} vs ZAR {mean_b:,.2f}. "
                f"These postal codes have materially different profit contributions — "
                f"consider differential premium adjustment."
            )
        else:
            interpretation = (
                f"Zip {code_a} vs {code_b}: FAIL TO REJECT H₀ (p = {result['p_value']:.4f}). "
                f"No significant margin difference at α = {alpha}."
            )

        results.append({
            "hypothesis": "H₃: No margin difference across zip codes",
            "group_a": str(code_a),
            "group_b": str(code_b),
            "kpi": "Margin (ZAR)",
            "test_used": "Welch's t-test",
            "mean_a": round(mean_a, 4),
            "mean_b": round(mean_b, 4),
            **result,
            "interpretation": interpretation,
        })
    return results


# ---------------------------------------------------------------------------
# H4 — Risk difference between genders
# ---------------------------------------------------------------------------

def risk_diff_gender(
    df: pd.DataFrame,
    kpi: str = "ClaimFrequency",
    alpha: float = 0.05,
) -> dict:
    """
    H₀: There is no significant risk difference between Women and Men.

    Parameters
    ----------
    kpi : 'ClaimFrequency' | 'ClaimSeverity'
    """
    df = _ensure_metrics(df)
    df_gender = df[df["Gender"].isin(["Female", "Male"])].copy()

    if kpi == "ClaimFrequency":
        a = df_gender.loc[df_gender["Gender"] == "Male",   "HasClaim"]
        b = df_gender.loc[df_gender["Gender"] == "Female", "HasClaim"]
        result = _ztest(a, b, alpha)
        test_name = "Two-proportion z-test"
        kpi_label = "Claim Frequency"
    else:
        a = df_gender.loc[(df_gender["Gender"] == "Male")   & (df_gender["HasClaim"] == 1), "TotalClaims"]
        b = df_gender.loc[(df_gender["Gender"] == "Female") & (df_gender["HasClaim"] == 1), "TotalClaims"]
        result = _welch_ttest(a, b, alpha)
        test_name = "Welch's t-test"
        kpi_label = "Claim Severity (ZAR)"

    mean_male   = a.mean()
    mean_female = b.mean()

    if result["decision"] == "Reject H₀":
        higher = "Male" if mean_male > mean_female else "Female"
        pct = abs((mean_male - mean_female) / mean_female * 100)
        interpretation = (
            f"We REJECT H₀ for gender (p = {result['p_value']:.4f} < {alpha}). "
            f"{higher} policyholders show higher {kpi_label} by ~{pct:.1f}%. "
            f"Note: any gender-based pricing must comply with SA short-term insurance "
            f"regulations and the Treating Customers Fairly (TCF) framework."
        )
    else:
        interpretation = (
            f"We FAIL TO REJECT H₀ for gender (p = {result['p_value']:.4f} ≥ {alpha}). "
            f"No statistically significant {kpi_label} difference between "
            f"Male (mean={mean_male:,.2f}) and Female (mean={mean_female:,.2f}) policyholders."
        )

    return {
        "hypothesis": "H₄: No risk difference between Women and Men",
        "group_a": "Male",
        "group_b": "Female",
        "kpi": kpi_label,
        "test_used": test_name,
        "mean_a": round(mean_male,   4),
        "mean_b": round(mean_female, 4),
        **result,
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# Summary table builder
# ---------------------------------------------------------------------------

def build_results_table(results: list[dict]) -> pd.DataFrame:
    """
    Accepts a flat list of result dicts and returns a formatted DataFrame
    suitable for display in a notebook or report.
    """
    rows = []
    for r in results:
        rows.append({
            "Hypothesis":     r.get("hypothesis", ""),
            "KPI":            r.get("kpi", ""),
            "Group A":        r.get("group_a", ""),
            "Group B":        r.get("group_b", ""),
            "Test":           r.get("test_used", ""),
            "Statistic":      r.get("statistic", ""),
            "p-value":        r.get("p_value", ""),
            "Decision":       r.get("decision", ""),
            "Interpretation": r.get("interpretation", ""),
        })
    return pd.DataFrame(rows)