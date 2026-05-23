import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns


# ── Palette / style ──────────────────────────────────────────────────────────
PALETTE = "Blues_d"
sns.set_theme(style="whitegrid", palette="muted")


# ── Summarisation ─────────────────────────────────────────────────────────────

def summarize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics (numerical + categorical) with dtype column."""
    stats = df.describe(include="all").T
    stats.insert(0, "dtype", df.dtypes)
    return stats


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Sorted table of columns that have at least one missing value."""
    n_missing = df.isnull().sum()
    pct_missing = 100 * n_missing / len(df)
    report = pd.DataFrame({"n_missing": n_missing, "pct_missing": pct_missing})
    return report[report["n_missing"] > 0].sort_values("pct_missing", ascending=False)


def dtype_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Column-level dtype and cardinality overview."""
    rows = []
    for col in df.columns:
        rows.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "n_unique": df[col].nunique(),
            "n_missing": df[col].isnull().sum(),
            "sample_values": str(df[col].dropna().unique()[:3].tolist()),
        })
    return pd.DataFrame(rows).set_index("column")


# ── Univariate plots ──────────────────────────────────────────────────────────

def plot_numerical_distributions(
    df: pd.DataFrame,
    cols: list[str],
    figsize: tuple = (18, 12),
    bins: int = 50,
) -> plt.Figure:
    """Grid of histograms for numerical columns."""
    ncols = 3
    nrows = (len(cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    for i, col in enumerate(cols):
        data = df[col].dropna()
        axes[i].hist(data, bins=bins, color="steelblue", edgecolor="white", linewidth=0.4)
        axes[i].set_title(col, fontsize=10, fontweight="bold")
        axes[i].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        axes[i].tick_params(axis="x", rotation=30, labelsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Numerical Feature Distributions", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


def plot_categorical_bar(
    df: pd.DataFrame,
    col: str,
    top_n: int = 15,
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """Horizontal bar chart of the top-N most frequent categories."""
    counts = df[col].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=figsize)
    counts.sort_values().plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(f"Top {top_n} – {col}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Policy Count")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.tight_layout()
    return fig


# ── Bivariate / multivariate ──────────────────────────────────────────────────

def plot_correlation_matrix(
    df: pd.DataFrame,
    cols: list[str],
    figsize: tuple = (12, 10),
) -> plt.Figure:
    """Lower-triangle correlation heatmap."""
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, linewidths=0.5, ax=ax, annot_kws={"size": 9},
    )
    ax.set_title("Correlation Matrix – Key Numerical Features", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_premium_vs_claims_scatter(
    df: pd.DataFrame,
    hue_col: str | None = None,
    sample_n: int = 10_000,
    figsize: tuple = (10, 7),
) -> plt.Figure:
    """Scatter plot of TotalPremium vs TotalClaims (sampled for speed)."""
    plot_df = df.sample(min(sample_n, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=figsize)
    scatter_kw = dict(alpha=0.4, s=10, linewidths=0)
    if hue_col:
        groups = plot_df[hue_col].cat.categories if hasattr(plot_df[hue_col], "cat") else plot_df[hue_col].unique()
        palette = sns.color_palette("tab10", n_colors=len(groups))
        for color, grp in zip(palette, groups):
            mask = plot_df[hue_col] == grp
            ax.scatter(plot_df.loc[mask, "TotalPremium"], plot_df.loc[mask, "TotalClaims"],
                       label=grp, color=color, **scatter_kw)
        ax.legend(title=hue_col, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    else:
        ax.scatter(plot_df["TotalPremium"], plot_df["TotalClaims"],
                   color="steelblue", **scatter_kw)
    ax.set_xlabel("Total Premium (ZAR)")
    ax.set_ylabel("Total Claims (ZAR)")
    ax.set_title("Total Premium vs Total Claims", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.tight_layout()
    return fig


# ── Outlier detection ─────────────────────────────────────────────────────────

def plot_boxplots(
    df: pd.DataFrame,
    cols: list[str],
    figsize: tuple = (18, 5),
) -> plt.Figure:
    """Side-by-side box plots for outlier detection."""
    fig, axes = plt.subplots(1, len(cols), figsize=figsize)
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        data = df[col].dropna()
        ax.boxplot(data, patch_artist=True,
                   boxprops=dict(facecolor="steelblue", alpha=0.6),
                   medianprops=dict(color="red", linewidth=2),
                   flierprops=dict(marker=".", markersize=2, alpha=0.3))
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.suptitle("Outlier Detection – Box Plots", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def outlier_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """IQR-based outlier count and percentage for each column."""
    rows = []
    for col in cols:
        data = df[col].dropna()
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        iqr = q3 - q1
        n_outliers = ((data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)).sum()
        rows.append({
            "column": col, "Q1": q1, "Q3": q3, "IQR": iqr,
            "lower_fence": q1 - 1.5 * iqr, "upper_fence": q3 + 1.5 * iqr,
            "n_outliers": n_outliers, "pct_outliers": 100 * n_outliers / len(data),
        })
    return pd.DataFrame(rows).set_index("column")


# ── Aggregation helpers ───────────────────────────────────────────────────────

def loss_ratio_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Portfolio-level loss ratio and margin aggregated by a categorical column."""
    return (
        df.groupby(group_col, observed=True)
        .agg(
            TotalPremium=("TotalPremium", "sum"),
            TotalClaims=("TotalClaims", "sum"),
            PolicyCount=("PolicyID", "count"),
            AvgPremium=("TotalPremium", "mean"),
        )
        .assign(
            LossRatio=lambda x: x["TotalClaims"] / x["TotalPremium"],
            Margin=lambda x: x["TotalPremium"] - x["TotalClaims"],
        )
        .sort_values("LossRatio", ascending=False)
    )


def claim_stats_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Claim frequency and severity by group."""
    grp = df.groupby(group_col, observed=True)
    freq = grp["HasClaim"].mean().rename("ClaimFrequency")
    severity = df[df["HasClaim"] == 1].groupby(group_col, observed=True)["TotalClaims"].mean().rename("ClaimSeverity")
    return pd.concat([freq, severity], axis=1).sort_values("ClaimFrequency", ascending=False)
