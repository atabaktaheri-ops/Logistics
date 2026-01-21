from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# Optional models (guarded imports)
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

# We intentionally do NOT use CatBoost in this app, because of sklearn tag incompatibilities
# If you later want CatBoost, we can add version pinning. For now: stability > everything.


# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent
EXCEL_PATH = BASE_DIR / "test.xlsx"
RANDOM_STATE = 42


# =========================
# FIX: Mixed int/str categorical
# =========================
class ToStringTransformer(BaseEstimator, TransformerMixin):
    """Convert categorical values to strings to avoid OneHotEncoder mixed-type crashes."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        return X_df.applymap(lambda v: v if pd.isna(v) else str(v)).values


# =========================
# HELPERS
# =========================
def make_monthyear(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "MonthYear" in df.columns:
        return df

    for c in df.columns:
        cl = c.lower()
        if "date" in cl or "month" in cl:
            dt = pd.to_datetime(df[c], errors="coerce")
            if dt.notna().any():
                df["MonthYear"] = dt.dt.to_period("M").astype(str)
                break
    return df


def safe_stratify(y: np.ndarray, max_classes: int = 20) -> Optional[np.ndarray]:
    uniq, cnt = np.unique(y, return_counts=True)
    if len(uniq) <= max_classes and np.all(cnt >= 2):
        return y
    return None


def build_preprocessor(cat_cols: List[str], num_cols: List[str]) -> ColumnTransformer:
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("to_str", ToStringTransformer()),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, cat_cols),
            ("num", num_pipe, num_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def get_numeric_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def safe_predict_proba_pos(estimator: Any, X) -> Optional[np.ndarray]:
    """
    Safely return P(y==1). If predict_proba isn't available or only one class exists, return None.
    Handles the case where predict_proba returns shape (n,1).
    """
    if not hasattr(estimator, "predict_proba"):
        return None
    proba = estimator.predict_proba(X)
    if proba is None:
        return None
    if proba.ndim != 2:
        return None
    if proba.shape[1] < 2:
        # only one class present in training
        return None

    # Find which column corresponds to class "1"
    try:
        classes = estimator.classes_
        if 1 in classes:
            idx = int(np.where(classes == 1)[0][0])
            return proba[:, idx]
        # Otherwise fallback to last column
        return proba[:, -1]
    except Exception:
        return proba[:, -1]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob_pos: Optional[np.ndarray]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    if y_prob_pos is not None:
        try:
            # roc_auc requires both classes present in y_true
            if len(np.unique(y_true)) < 2:
                out["roc_auc"] = None
            else:
                out["roc_auc"] = float(roc_auc_score(y_true, y_prob_pos))
        except Exception:
            out["roc_auc"] = None
    else:
        out["roc_auc"] = None
    return out


def best_by_metric(results: Dict[str, "ModelResult"], primary: str) -> "ModelResult":
    def score(r: ModelResult) -> float:
        v = r.metrics.get(primary)
        if v is None:
            v = r.metrics.get("f1")
        if v is None:
            v = r.metrics.get("accuracy")
        return float(v) if v is not None else -1e9

    return max(results.values(), key=score)


# =========================
# CACHING: LOAD DATA (mtime-based)
# =========================
@st.cache_data(show_spinner=False)
def load_excel_cached(excel_path: str, file_mtime: float, debug: bool) -> pd.DataFrame:
    if debug:
        st.write("🧪 DEBUG: Reading Excel now (only on cold start / file change).")
    df = pd.read_excel(excel_path)
    return make_monthyear(df)


# =========================
# TRAINING (cached by file + settings)
# =========================
@dataclass
class ModelResult:
    name: str
    pipeline: Any
    best_params: Dict[str, Any]
    metrics: Dict[str, Optional[float]]
    y_pred: np.ndarray
    y_prob_pos: Optional[np.ndarray]


@st.cache_resource(show_spinner=True)
def train_all_models_cached(
    excel_path: str,
    file_mtime: float,
    debug: bool,
    enable_shap: bool,
    max_rows: Optional[int],
    test_size: float,
    primary_metric: str,
    target_mode: str,
    target_column: str,
):
    df = load_excel_cached(excel_path, file_mtime, debug)

    if max_rows is not None and max_rows > 0 and len(df) > max_rows:
        df = df.sample(max_rows, random_state=RANDOM_STATE).reset_index(drop=True)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    # Target creation with correct alignment
    if target_mode == "Binary sign (>=0) from numeric column":
        gp = pd.to_numeric(df[target_column], errors="coerce")
        df = df[gp.notna()].copy()
        y = (gp.loc[df.index] >= 0).astype(int).values
        X = df.drop(columns=[target_column], errors="ignore")
        target_desc = f"{target_column} (sign)"
    else:
        y_raw = pd.to_numeric(df[target_column], errors="coerce")
        df = df[y_raw.notna()].copy()
        y = y_raw.loc[df.index].astype(int).values
        uniq = np.unique(y)
        if len(uniq) > 20:
            raise ValueError(
                f"Label column '{target_column}' has {len(uniq)} unique values; not a classification label. "
                "Use Binary sign mode instead."
            )
        X = df.drop(columns=[target_column], errors="ignore")
        target_desc = target_column

    if len(X) < 20:
        raise ValueError(f"Too few usable rows after cleaning: {len(X)}.")

    # Prepare preprocessor
    cat_cols = [c for c in X.columns if (X[c].dtype == "object" or str(X[c].dtype).startswith("category"))]
    num_cols = [c for c in X.columns if c not in cat_cols]
    pre = build_preprocessor(cat_cols, num_cols)

    # Split safely
    strat = safe_stratify(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=strat
    )

    unique_train = np.unique(y_train)
    single_class_train = (len(unique_train) < 2)

    # Always include a baseline that never fails
    models: Dict[str, Any] = {
        "Dummy (most_frequent)": DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1, class_weight=None
        ),
    }

    # Only include models that require 2 classes if we actually have 2 classes
    if not single_class_train:
        models["AdaBoost"] = AdaBoostClassifier(n_estimators=250, learning_rate=0.6, random_state=RANDOM_STATE)
        models["KernelSVM"] = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)
        if XGBClassifier is not None:
            models["XGBoost"] = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.06,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                n_jobs=-1,
            )

    # Light grids (optional). For stability, keep very small.
    grids: Dict[str, Dict[str, List[Any]]] = {
        "RandomForest": {"clf__n_estimators": [300, 500], "clf__max_depth": [None, 12]},
    }
    if "AdaBoost" in models:
        grids["AdaBoost"] = {"clf__n_estimators": [150, 300], "clf__learning_rate": [0.3, 0.8]}
    if "KernelSVM" in models:
        grids["KernelSVM"] = {"clf__C": [0.7, 1.5], "clf__gamma": ["scale", "auto"]}
    if "XGBoost" in models:
        grids["XGBoost"] = {"clf__max_depth": [4, 7], "clf__learning_rate": [0.05, 0.1]}

    results: Dict[str, ModelResult] = {}
    errors: Dict[str, str] = {}

    for name, clf in models.items():
        grid = grids.get(name, {})
        param_list = list(ParameterGrid(grid)) if grid else [None]

        best_score = -1e18
        best_pipe = None
        best_params = {}
        best_pred = None
        best_prob = None
        best_metrics = None

        for params in param_list:
            try:
                pipe = Pipeline([("pre", pre), ("clf", clf)])
                if params:
                    pipe.set_params(**params)
                pipe.fit(X_train, y_train)

                y_pred = pipe.predict(X_test)
                y_prob_pos = safe_predict_proba_pos(pipe, X_test)  # safe for single-class too
                m = compute_metrics(y_test, y_pred, y_prob_pos)

                s = m.get(primary_metric) or m.get("f1") or m.get("accuracy")
                s_val = float(s) if s is not None else -1e9

                if s_val > best_score:
                    best_score = s_val
                    best_pipe = pipe
                    best_params = params or {}
                    best_pred = y_pred
                    best_prob = y_prob_pos
                    best_metrics = m

            except Exception as e:
                errors[name] = str(e)
                continue

        if best_pipe is not None and best_pred is not None and best_metrics is not None:
            results[name] = ModelResult(
                name=name,
                pipeline=best_pipe,
                best_params=best_params,
                metrics=best_metrics,
                y_pred=best_pred,
                y_prob_pos=best_prob,
            )

    if not results:
        # Now we provide the real errors so you can diagnose.
        raise RuntimeError(
            "All models failed to train. Errors: " + " | ".join([f"{k}: {v}" for k, v in errors.items()])[:1500]
        )

    best = best_by_metric(results, primary_metric)

    # Optional SHAP (only if we have predict_proba and not single-class)
    shap_top = None
    shap_err = None
    if enable_shap:
        try:
            if best.y_prob_pos is None:
                raise RuntimeError("Best model has no usable predict_proba; SHAP disabled.")
            if len(np.unique(y_train)) < 2:
                raise RuntimeError("Training data has only one class; SHAP is not meaningful.")

            import shap  # lazy

            X_test_trans = best.pipeline.named_steps["pre"].transform(X_test)
            n_bg = min(100, X_test_trans.shape[0])
            n_sub = min(80, X_test_trans.shape[0])

            X_bg = shap.sample(X_test_trans, n_bg, random_state=RANDOM_STATE)
            X_sub = shap.sample(X_test_trans, n_sub, random_state=RANDOM_STATE)

            # Extract classifier only (post-preprocessing)
            clf = best.pipeline.named_steps["clf"]

            def pred_pos(x):
                proba = clf.predict_proba(x)
                if proba.shape[1] < 2:
                    return np.zeros(proba.shape[0])
                # find column for class 1
                if hasattr(clf, "classes_") and 1 in clf.classes_:
                    idx = int(np.where(clf.classes_ == 1)[0][0])
                    return proba[:, idx]
                return proba[:, -1]

            explainer = shap.KernelExplainer(pred_pos, X_bg)
            sv = explainer.shap_values(X_sub, nsamples=120)
            if isinstance(sv, list) and len(sv) == 2:
                sv = sv[1]
            sv = np.asarray(sv)

            mean_abs = np.mean(np.abs(sv), axis=0)
            shap_top = (
                pd.DataFrame({"feature": [f"f{i}" for i in range(len(mean_abs))], "mean_abs_shap": mean_abs})
                .sort_values("mean_abs_shap", ascending=False)
                .head(25)
            )
        except Exception as e:
            shap_err = f"SHAP failed: {e}"

    return {
        "target_desc": target_desc,
        "results": results,
        "best": best,
        "y_test": y_test,
        "single_class_train": single_class_train,
        "errors": errors,
        "shap_top": shap_top,
        "shap_err": shap_err,
    }


# =========================
# UI
# =========================
def main():
    st.set_page_config(page_title="Air Cargo GP Classifier", layout="wide")
    st.title("Air Cargo GP Classifier (Robust + Cloud-ready)")

    if not EXCEL_PATH.exists():
        st.error(f"Missing {EXCEL_PATH.name}. Put Test.xlsx next to app.py.")
        st.stop()

    mtime = EXCEL_PATH.stat().st_mtime

    # Preview for dropdowns
    df_preview = load_excel_cached(str(EXCEL_PATH), mtime, debug=False)
    all_cols = list(df_preview.columns)
    numeric_cols = get_numeric_cols(df_preview)

    with st.sidebar:
        st.header("Settings")
        debug = st.checkbox("Debug mode", value=False)
        enable_shap = st.checkbox("Compute SHAP (slow)", value=False)

        st.markdown("---")
        st.subheader("Target")

        target_mode = st.selectbox(
            "Target mode",
            ["Binary sign (>=0) from numeric column", "Use existing binary label column"],
            index=0,
        )

        if target_mode == "Binary sign (>=0) from numeric column":
            if numeric_cols:
                # try to guess likely GP column
                default_idx = 0
                for guess in ["gp", "GP", "gross_profit", "GrossProfit", "profit", "Profit"]:
                    if guess in numeric_cols:
                        default_idx = numeric_cols.index(guess)
                        break
                target_column = st.selectbox("Numeric GP column", numeric_cols, index=default_idx)
            else:
                target_column = st.selectbox("Column (no numeric detected)", all_cols, index=0)
        else:
            default_idx = all_cols.index("gp_sign") if "gp_sign" in all_cols else 0
            target_column = st.selectbox("Label column", all_cols, index=default_idx)

        st.markdown("---")
        st.subheader("Training")
        test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
        primary_metric = st.selectbox("Primary metric", ["roc_auc", "f1", "accuracy"], index=0)

        max_rows = st.number_input("Max rows (0 = use all)", min_value=0, value=0, step=1000)
        max_rows_opt = None if int(max_rows) == 0 else int(max_rows)

        st.markdown("---")
        if st.button("Clear Streamlit cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cache cleared. Re-run will reload/retrain as needed.")

    with st.spinner("Training (cached)…"):
        pack = train_all_models_cached(
            excel_path=str(EXCEL_PATH),
            file_mtime=mtime,
            debug=debug,
            enable_shap=enable_shap,
            max_rows=max_rows_opt,
            test_size=float(test_size),
            primary_metric=str(primary_metric),
            target_mode=str(target_mode),
            target_column=str(target_column),
        )

    if pack["single_class_train"]:
        st.warning(
            "⚠️ Training data contains only ONE class after cleaning (e.g., all GP>=0). "
            "Some models are automatically disabled; using baseline + RandomForest."
        )

    # Summary
    results: Dict[str, ModelResult] = pack["results"]
    best: ModelResult = pack["best"]
    y_test = pack["y_test"]

    st.subheader("Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Target", pack["target_desc"])
    c2.metric("Best model", best.name)
    c3.metric("ROC-AUC", f"{best.metrics.get('roc_auc'):.3f}" if best.metrics.get("roc_auc") is not None else "NA")
    c4.metric("F1", f"{best.metrics.get('f1'):.3f}" if best.metrics.get("f1") is not None else "NA")

    st.subheader("Model comparison (test)")
    rows = []
    for name, r in results.items():
        rows.append(
            {
                "Model": name,
                "ROC_AUC": r.metrics.get("roc_auc"),
                "F1": r.metrics.get("f1"),
                "Accuracy": r.metrics.get("accuracy"),
                "Precision": r.metrics.get("precision"),
                "Recall": r.metrics.get("recall"),
            }
        )
    st.dataframe(pd.DataFrame(rows).sort_values(by="ROC_AUC", ascending=False), use_container_width=True)

    st.subheader(f"Confusion matrix — {best.name}")
    cm = confusion_matrix(y_test, best.y_pred)
    fig = px.imshow(cm, text_auto=True, title=f"Confusion Matrix — {best.name}", labels=dict(x="Pred", y="True"))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Best hyperparameters"):
        st.write(best.best_params)

    # Show training errors (super useful)
    with st.expander("Model training errors (if any)"):
        if not pack["errors"]:
            st.write("No errors.")
        else:
            st.json(pack["errors"])

    # SHAP
    if enable_shap:
        st.subheader("SHAP (optional)")
        if pack["shap_err"]:
            st.warning(pack["shap_err"])
        elif pack["shap_top"] is not None:
            st.dataframe(pack["shap_top"], use_container_width=True)
            fig2 = px.bar(pack["shap_top"], x="mean_abs_shap", y="feature", orientation="h",
                          title="Top SHAP features (generic indices)")
            st.plotly_chart(fig2, use_container_width=True)

    st.caption(f"Excel file: {EXCEL_PATH.name} | mtime: {mtime}")


if __name__ == "__main__":
    main()
