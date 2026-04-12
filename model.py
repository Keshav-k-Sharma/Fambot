# =========================
# 1. IMPORTS
# =========================
import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    make_scorer,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

_ROOT = Path(__file__).resolve().parent
_DATA_CSV = _ROOT / "sources" / "diabetes.csv"
_MODEL_PATH = _ROOT / "diabetes_model.pkl"
_PLOT_PATH = _ROOT / "feature_importance.png"

# Columns where 0 is invalid (Pima Indians heuristic)
_ZERO_INVALID_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


class ZeroToNaNTransformer(BaseEstimator, TransformerMixin):
    """Replace zeros with NaN for the selected medical columns (invalid sentinel)."""

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "copy"):
            out = X.copy()
            for col in self.columns:
                if col in out.columns:
                    out[col] = out[col].replace(0, np.nan)
            return out
        arr = np.asarray(X, dtype=float).copy()
        arr[arr == 0] = np.nan
        return arr


def _feature_names_in_order(X: pd.DataFrame) -> list[str]:
    """Match ColumnTransformer output order: imputed medical cols, then passthrough rest."""
    zero_cols = [c for c in _ZERO_INVALID_COLS if c in X.columns]
    other_cols = [c for c in X.columns if c not in zero_cols]
    return zero_cols + other_cols


def _build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    zero_cols = [c for c in _ZERO_INVALID_COLS if c in X.columns]
    other_cols = [c for c in X.columns if c not in zero_cols]
    transformers = [
        (
            "medical",
            Pipeline(
                [
                    ("z2n", ZeroToNaNTransformer(zero_cols)),
                    ("imp", SimpleImputer(strategy="median")),
                ]
            ),
            zero_cols,
        ),
    ]
    if other_cols:
        transformers.append(("rest", "passthrough", other_cols))
    return ColumnTransformer(
        transformers, remainder="drop", verbose_feature_names_out=False
    )


def main() -> None:
    # =========================
    # 2. LOAD DATA
    # =========================
    df = pd.read_csv(_DATA_CSV)

    print("Initial shape:", df.shape)
    print(df.head())

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    preprocess = _build_preprocess(X)

    # =========================
    # 3. STRATIFIED TRAIN / TEST (preprocessing fit only inside CV / pipeline)
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "roc_auc": "roc_auc",
        "f1_pos": make_scorer(f1_score, pos_label=1),
        "accuracy": "accuracy",
    }

    # =========================
    # 4. LOGISTIC REGRESSION PIPELINE + CV
    # =========================
    lr_pipe = Pipeline(
        [
            ("pre", preprocess),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    print("\n--- Logistic Regression (5-fold CV on train) ---")
    lr_cv = cross_validate(
        lr_pipe,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )
    for name, key in [("ROC-AUC", "test_roc_auc"), ("F1 (class 1)", "test_f1_pos"), ("Accuracy", "test_accuracy")]:
        scores = lr_cv[key]
        print(f"  {name}: {scores.mean():.4f} (+/- {scores.std():.4f})")

    lr_pipe.fit(X_train, y_train)
    y_pred_lr = lr_pipe.predict(X_test)
    y_proba_lr = lr_pipe.predict_proba(X_test)[:, 1]
    print("\n--- Logistic Regression (holdout) ---")
    print("  ROC-AUC:", roc_auc_score(y_test, y_proba_lr))
    print("  Accuracy:", accuracy_score(y_test, y_pred_lr))
    print("  F1 (class 1):", f1_score(y_test, y_pred_lr, pos_label=1))
    print(classification_report(y_test, y_pred_lr))

    lr_cv_auc = lr_cv["test_roc_auc"].mean()

    # =========================
    # 5. XGBOOST: RANDOM SEARCH + CV
    # =========================
    preprocess_xgb = _build_preprocess(X)
    xgb_pipe = Pipeline(
        [
            ("pre", preprocess_xgb),
            (
                "clf",
                XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    param_distributions = {
        "clf__max_depth": [3, 4, 5, 6, 8],
        "clf__n_estimators": [50, 100, 200, 300],
        "clf__learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
        "clf__min_child_weight": [1, 2, 3, 5],
        "clf__subsample": [0.7, 0.8, 0.9, 1.0],
        "clf__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "clf__reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }

    print("\n--- XGBoost (RandomizedSearchCV, 5-fold, roc_auc) ---")
    xgb_search = RandomizedSearchCV(
        estimator=xgb_pipe,
        param_distributions=param_distributions,
        n_iter=30,
        cv=cv,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1,
        refit=True,
    )
    xgb_search.fit(X_train, y_train)
    print(f"  Best CV ROC-AUC: {xgb_search.best_score_:.4f}")
    print(f"  Best params: {xgb_search.best_params_}")

    best_xgb = xgb_search.best_estimator_
    y_pred_xgb = best_xgb.predict(X_test)
    y_proba_xgb = best_xgb.predict_proba(X_test)[:, 1]

    print("\n--- XGBoost (holdout) ---")
    print("  ROC-AUC:", roc_auc_score(y_test, y_proba_xgb))
    print("  Accuracy:", accuracy_score(y_test, y_pred_xgb))
    print("  F1 (class 1):", f1_score(y_test, y_pred_xgb, pos_label=1))
    print(classification_report(y_test, y_pred_xgb))

    xgb_cv_auc = xgb_search.best_score_

    # =========================
    # 6. SAVE BEST PIPELINE (by CV ROC-AUC) + FEATURE IMPORTANCE
    # =========================
    import joblib

    feat_names = _feature_names_in_order(X)

    if xgb_cv_auc >= lr_cv_auc:
        champion = best_xgb
        champion_name = "XGBoost (RandomizedSearchCV best)"
        importances = champion.named_steps["clf"].feature_importances_
        plt.figure()
        plt.barh(feat_names, importances)
        plt.title("Feature Importance (XGBoost)")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.savefig(_PLOT_PATH)
        plt.close()
        print(f"\nChampion (CV ROC-AUC): {champion_name}")
        print(f"Feature importance plot saved to {_PLOT_PATH}")
    else:
        champion = lr_pipe
        champion_name = "LogisticRegression"
        coef = champion.named_steps["clf"].coef_.ravel()
        plt.figure()
        plt.barh(feat_names, np.abs(coef))
        plt.title("Absolute coefficients (Logistic Regression)")
        plt.xlabel("|Coefficient|")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.savefig(_PLOT_PATH)
        plt.close()
        print(f"\nChampion (CV ROC-AUC): {champion_name}")
        print(f"Coefficient magnitude plot saved to {_PLOT_PATH}")

    joblib.dump(champion, _MODEL_PATH)
    print(f"\nSaved pipeline ({champion_name}) to {_MODEL_PATH}")


if __name__ == "__main__":
    main()
