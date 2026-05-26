"""
src/modeling.py
---------------
Reusable modeling pipeline for ACIS risk-based pricing (Task 4).

Models
------
- Linear Regression  (log-transformed target, interpretable baseline)
- Random Forest      (non-linear interactions)
- XGBoost            (production candidate)

Two tasks
---------
1. Claim Severity   – regression on TotalClaims > 0, target = log1p(TotalClaims)
2. Claim Probability – binary classification (HasClaim), used with severity for pricing

Usage
-----
    from src.modeling import (
        prepare_features, train_severity_models,
        train_frequency_models, evaluate_regression,
        evaluate_classification, compute_risk_premium,
        plot_shap_summary
    )
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model  import LinearRegression, LogisticRegression
from sklearn.ensemble       import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics        import (
    mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from sklearn.preprocessing  import LabelEncoder
import joblib

try:
    from xgboost import XGBRegressor, XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    warnings.warn("xgboost not installed; XGBoost models will be skipped.")

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    warnings.warn("shap not installed; SHAP plots will be skipped.")


CATEGORICAL_COLS = [
    "Province", "Gender", "VehicleType", "Make", "BodyType",
    "CoverType", "CoverGroup", "MaritalStatus", "LegalType",
]

NUMERIC_COLS = [
    "VehicleAge", "Kilowatts", "Cubiccapacity", "SumInsured",
    "CalculatedPremiumPerTerm", "ExcessSelected", "CustomValueEstimate",
    "CapitalOutstanding",
]

DROP_COLS = [
    "UnderwrittenCoverID", "PolicyID", "TransactionMonth",
    "TotalPremium", "TotalClaims", "LossRatio", "Margin", "HasClaim",
    "NumberOfVehiclesInFleet", "CrossBorder",
]


def prepare_features(
    df,
    target="TotalClaims",
    task="severity",
    test_size=0.2,
    random_state=42,
):
    df = df.copy()
    if "VehicleAge" not in df.columns:
        df["VehicleAge"] = 2015 - pd.to_numeric(
            df.get("RegistrationYear", 2010), errors="coerce"
        ).fillna(2010)

    if task == "severity":
        df = df[df["TotalClaims"] > 0].copy()
        y = np.log1p(df["TotalClaims"])
    else:
        y = (df["TotalClaims"] > 0).astype(int)

    drop = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=drop, errors="ignore")

    le = LabelEncoder()
    for col in CATEGORICAL_COLS:
        if col in X.columns:
            X[col] = X[col].astype(str).fillna("Unknown")
            X[col] = le.fit_transform(X[col])

    X = X.select_dtypes(include=[np.number])
    X = X.fillna(X.median())

    feature_names = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, feature_names


def train_severity_models(X_train, y_train, random_state=42):
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=10,
            min_samples_leaf=5, n_jobs=-1,
            random_state=random_state,
        ),
    }
    if _XGB_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300, max_depth=6,
            learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=random_state,
            verbosity=0,
        )
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        print(f"  ✓ {name} trained")
    return fitted


def train_frequency_models(X_train, y_train, random_state=42):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=500, random_state=random_state),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=8,
            min_samples_leaf=10, n_jobs=-1,
            random_state=random_state,
        ),
    }
    if _XGB_AVAILABLE:
        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        scale = neg / pos if pos > 0 else 1
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=5,
            learning_rate=0.05, scale_pos_weight=scale,
            random_state=random_state, verbosity=0,
            eval_metric="logloss",
        )
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        print(f"  ✓ {name} trained")
    return fitted


def evaluate_regression(models, X_test, y_test):
    rows = []
    for name, model in models.items():
        y_pred_log = model.predict(X_test)
        y_pred     = np.expm1(y_pred_log)
        y_true     = np.expm1(y_test)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)
        rows.append({"Model": name, "RMSE (ZAR)": round(rmse, 2), "R²": round(r2, 4)})
    return pd.DataFrame(rows).sort_values("RMSE (ZAR)")


def evaluate_classification(models, X_test, y_test):
    rows = []
    for name, model in models.items():
        y_pred  = model.predict(X_test)
        y_proba = (model.predict_proba(X_test)[:, 1]
                   if hasattr(model, "predict_proba") else y_pred)
        rows.append({
            "Model":     name,
            "Accuracy":  round(accuracy_score(y_test, y_pred),  4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall":    round(recall_score(y_test, y_pred, zero_division=0),    4),
            "F1":        round(f1_score(y_test, y_pred, zero_division=0),        4),
            "ROC-AUC":   round(roc_auc_score(y_test, y_proba),                  4),
        })
    return pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)


def compute_risk_premium(
    prob_claim,
    predicted_severity,
    expense_loading=0.15,
    profit_margin=0.05,
):
    pure_premium   = prob_claim * predicted_severity
    loaded_premium = pure_premium * (1 + expense_loading + profit_margin)
    return round(loaded_premium, 2)


def plot_shap_summary(model, X_test, feature_names, model_name="Best Model",
                      max_display=15, save_path=None):
    if not _SHAP_AVAILABLE:
        print("shap not installed. Run: pip install shap")
        return
    X_df = pd.DataFrame(X_test, columns=feature_names) if not isinstance(X_test, pd.DataFrame) else X_test
    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_df)
    except Exception:
        explainer   = shap.Explainer(model, X_df)
        shap_values = explainer(X_df).values
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_df, feature_names=feature_names,
                      max_display=max_display, show=False)
    plt.title(f"SHAP Feature Importance — {model_name}", fontsize=13, pad=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def get_shap_top_features(model, X_test, feature_names, top_n=10):
    if not _SHAP_AVAILABLE:
        print("shap not installed.")
        return pd.DataFrame()
    X_df = pd.DataFrame(X_test, columns=feature_names) if not isinstance(X_test, pd.DataFrame) else X_test
    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_df)
    except Exception:
        explainer   = shap.Explainer(model, X_df)
        shap_values = explainer(X_df).values
    mean_abs = np.abs(shap_values).mean(axis=0)
    return pd.DataFrame({
        "Feature":     feature_names,
        "Mean |SHAP|": np.round(mean_abs, 6),
    }).sort_values("Mean |SHAP|", ascending=False).head(top_n).reset_index(drop=True)


def save_model(model, path):
    joblib.dump(model, path)
    print(f"  Model saved → {path}")


def load_model(path):
    return joblib.load(path)
