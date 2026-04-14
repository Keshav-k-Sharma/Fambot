# =========================
# 1. IMPORTS
# =========================
import json
import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
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
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from fambot_backend.cardio_features import BASE_FEATURES, FEATURE_ORDER

warnings.filterwarnings("ignore", category=UserWarning)

_ROOT = Path(__file__).resolve().parent
_DATA_CSV = _ROOT / "sources" / "cardio_train.csv"
_MODEL_PATH = _ROOT / "cardiovascular_model.pkl"
_THRESHOLD_PATH = _ROOT / "cardiovascular_model.threshold.json"
_PLOT_PATH = _ROOT / "feature_importance.png"


def _clean_cardio_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Load target and features; engineer age_years and derived vitals; filter bad rows."""
    y = df["cardio"].astype(int)
    X = df.drop(columns=["cardio"]).copy()
    X["age_years"] = X["age"].astype(float) / 365.25
    X = X.drop(columns=["age"])
    X = X[BASE_FEATURES]

    ok = (
        (X["ap_hi"] > X["ap_lo"])
        & (X["ap_hi"] >= 80)
        & (X["ap_hi"] <= 250)
        & (X["ap_lo"] >= 40)
        & (X["ap_lo"] <= 150)
        & (X["height"] >= 120)
        & (X["height"] <= 220)
        & (X["weight"] >= 35)
        & (X["weight"] <= 250)
    )
    X = X.loc[ok].reset_index(drop=True)
    y = y.loc[ok].reset_index(drop=True)

    h_m = X["height"].astype(float) / 100.0
    X["bmi"] = X["weight"].astype(float) / (h_m * h_m)
    X["pulse_pressure"] = X["ap_hi"].astype(float) - X["ap_lo"].astype(float)
    X["map_approx"] = (X["ap_hi"].astype(float) + 2.0 * X["ap_lo"].astype(float)) / 3.0

    X = X[FEATURE_ORDER]
    return X, y


def _feature_names_in_order(X: pd.DataFrame) -> list[str]:
    """Column order matches ColumnTransformer (single block, median imputation)."""
    return list(X.columns)


def _build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    cols = list(X.columns)
    return ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _oof_positive_proba(
    estimator: Pipeline, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold
) -> np.ndarray:
    """Out-of-fold positive-class probabilities for threshold tuning (no test leakage)."""
    return cross_val_predict(
        clone(estimator), X, y, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]


def _best_threshold_accuracy(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    """Threshold in (0,1) maximizing accuracy on OOF predictions."""
    best_t, best_acc = 0.5, 0.0
    for t in np.linspace(0.01, 0.99, 99):
        pred = (proba >= t).astype(int)
        acc = accuracy_score(y_true, pred)
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t, best_acc


def main() -> None:
    # =========================
    # 2. LOAD DATA
    # =========================
    raw = pd.read_csv(_DATA_CSV, sep=";")
    print("Initial shape:", raw.shape)
    print(raw.head())

    df = raw.drop(columns=["id"], errors="ignore")
    X, y = _clean_cardio_xy(df)

    print(f"\nAfter cleaning: X={X.shape}, y={y.shape}")
    print(X.head())

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
    # 5. XGBOOST: RANDOM SEARCH + CV (hist, wider search)
    # =========================
    preprocess_xgb = _build_preprocess(X)
    xgb_pipe = Pipeline(
        [
            ("pre", preprocess_xgb),
            (
                "clf",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.1,
                    tree_method="hist",
                    max_bin=256,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    xgb_param_distributions = {
        "clf__max_depth": [3, 4, 5, 6, 8],
        "clf__n_estimators": [100, 200, 300, 500],
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15, 0.2],
        "clf__min_child_weight": [1, 2, 3, 5, 7],
        "clf__subsample": [0.7, 0.8, 0.9, 1.0],
        "clf__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "clf__reg_lambda": [0.5, 1.0, 2.0, 5.0],
        "clf__reg_alpha": [0.0, 0.01, 0.1, 0.5, 1.0],
        "clf__gamma": [0.0, 0.05, 0.1, 0.2],
        "clf__max_bin": [128, 256, 512],
    }

    print("\n--- XGBoost (RandomizedSearchCV, 5-fold, roc_auc) ---")
    xgb_search = RandomizedSearchCV(
        estimator=xgb_pipe,
        param_distributions=xgb_param_distributions,
        n_iter=80,
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
    # 6. HIST GRADIENT BOOSTING: RANDOM SEARCH + CV
    # =========================
    preprocess_hgb = _build_preprocess(X)
    hgb_pipe = Pipeline(
        [
            ("pre", preprocess_hgb),
            (
                "clf",
                HistGradientBoostingClassifier(
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.12,
                    n_iter_no_change=20,
                ),
            ),
        ]
    )

    hgb_param_distributions = {
        "clf__max_depth": [None, 3, 5, 7, 9],
        "clf__learning_rate": [0.01, 0.05, 0.08, 0.1, 0.15, 0.2],
        "clf__max_iter": [150, 250, 400, 600],
        "clf__min_samples_leaf": [10, 20, 40, 80, 120],
        "clf__l2_regularization": [0.0, 1e-4, 1e-2, 0.1, 1.0],
        "clf__max_leaf_nodes": [15, 31, 63, 127],
    }

    print("\n--- HistGradientBoosting (RandomizedSearchCV, 5-fold, roc_auc) ---")
    hgb_search = RandomizedSearchCV(
        estimator=hgb_pipe,
        param_distributions=hgb_param_distributions,
        n_iter=60,
        cv=cv,
        scoring="roc_auc",
        random_state=43,
        n_jobs=-1,
        refit=True,
    )
    hgb_search.fit(X_train, y_train)
    print(f"  Best CV ROC-AUC: {hgb_search.best_score_:.4f}")
    print(f"  Best params: {hgb_search.best_params_}")

    best_hgb = hgb_search.best_estimator_
    y_pred_hgb = best_hgb.predict(X_test)
    y_proba_hgb = best_hgb.predict_proba(X_test)[:, 1]

    print("\n--- HistGradientBoosting (holdout) ---")
    print("  ROC-AUC:", roc_auc_score(y_test, y_proba_hgb))
    print("  Accuracy:", accuracy_score(y_test, y_pred_hgb))
    print("  F1 (class 1):", f1_score(y_test, y_pred_hgb, pos_label=1))
    print(classification_report(y_test, y_pred_hgb))

    hgb_cv_auc = hgb_search.best_score_

    # =========================
    # 7. CHAMPION BY CV ROC-AUC + OOF THRESHOLD FOR ACCURACY
    # =========================
    # Compare three models by mean CV ROC-AUC (primary metric per plan).
    candidates: list[tuple[str, float, Pipeline]] = [
        ("LogisticRegression", lr_cv_auc, lr_pipe),
        ("XGBoost (RandomizedSearchCV best)", xgb_cv_auc, best_xgb),
        ("HistGradientBoosting (RandomizedSearchCV best)", hgb_cv_auc, best_hgb),
    ]
    champion_name, _, champion = max(candidates, key=lambda t: t[1])

    print("\n--- Champion (highest mean CV ROC-AUC) ---")
    print(f"  {champion_name}  (CV ROC-AUC={max(lr_cv_auc, xgb_cv_auc, hgb_cv_auc):.4f})")

    oof_proba = _oof_positive_proba(champion, X_train, y_train, cv)
    threshold, oof_acc = _best_threshold_accuracy(y_train.to_numpy(), oof_proba)
    print(f"\n  OOF accuracy-optimal threshold: {threshold:.4f} (OOF accuracy={oof_acc:.4f})")

    y_proba_test = champion.predict_proba(X_test)[:, 1]
    y_pred_default = champion.predict(X_test)
    y_pred_thr = (y_proba_test >= threshold).astype(int)

    print("\n--- Champion holdout (default 0.5 threshold) ---")
    print("  ROC-AUC:", roc_auc_score(y_test, y_proba_test))
    print("  Accuracy:", accuracy_score(y_test, y_pred_default))
    print("  F1 (class 1):", f1_score(y_test, y_pred_default, pos_label=1))

    print("\n--- Champion holdout (OOF-tuned threshold, evaluated once on test) ---")
    print("  Accuracy:", accuracy_score(y_test, y_pred_thr))
    print("  F1 (class 1):", f1_score(y_test, y_pred_thr, pos_label=1))
    print(classification_report(y_test, y_pred_thr))

    # =========================
    # 8. SAVE PIPELINE + THRESHOLD SIDEcar + FEATURE IMPORTANCE
    # =========================
    import joblib

    feat_names = _feature_names_in_order(X)

    clf = champion.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        plt.figure()
        plt.barh(feat_names, importances)
        plt.title(f"Feature Importance ({champion_name})")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.savefig(_PLOT_PATH)
        plt.close()
        print(f"\nFeature importance plot saved to {_PLOT_PATH}")
    elif hasattr(clf, "coef_"):
        coef = clf.coef_.ravel()
        plt.figure()
        plt.barh(feat_names, np.abs(coef))
        plt.title(f"Absolute coefficients ({champion_name})")
        plt.xlabel("|Coefficient|")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.savefig(_PLOT_PATH)
        plt.close()
        print(f"\nCoefficient plot saved to {_PLOT_PATH}")

    joblib.dump(champion, _MODEL_PATH)
    print(f"\nSaved pipeline ({champion_name}) to {_MODEL_PATH}")

    meta = {
        "champion": champion_name,
        "positive_class_threshold": threshold,
        "oof_accuracy_at_threshold": oof_acc,
        "cv_roc_auc_logistic_regression": lr_cv_auc,
        "cv_roc_auc_xgboost": xgb_cv_auc,
        "cv_roc_auc_hist_gradient_boosting": hgb_cv_auc,
    }
    _THRESHOLD_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved threshold metadata to {_THRESHOLD_PATH}")


if __name__ == "__main__":
    main()
