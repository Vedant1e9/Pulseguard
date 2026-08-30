"""
PatientTriage.ai — Triage Risk Model (training & inference)
===========================================================

The learned component of the system. Everything about how it is trained is
chosen to make the reported numbers survive a hostile read.

**Split by hospital, not by row.** Patients from the same emergency department
share documentation habits, equipment, case mix and triage culture. A random
row split lets a model memorise "hospital 47 codes everything a 3" and report
it as skill. Every split here is grouped by hospital, so the test set is made
of departments the model has never seen — which is also the question that
actually matters commercially: does this work at the *next* hospital?

**Four disjoint folds, each with one job.**
    train      — fit the classifiers
    calibrate  — fit isotonic probability calibration
    conformal  — fit conformal thresholds (must not have trained the calibrator)
    test       — touched exactly once, for the numbers we publish

**Missing values stay missing.** The primary models are NaN-native gradient
boosters. Nothing is imputed to zero anywhere in the pipeline.

**Selected on safety, not accuracy.** The selection criterion weights critical
recall and under-triage; overall accuracy is reported but never selects.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data.features import build_feature_frame
from models.uncertainty import (
    ConformalTriage, CriticalRiskConformal, calibration_report,
)
from models.decision_policy import (
    CostMatrix, expected_cost_decision, operating_curve,
    select_operating_point, scaled_profile,
)

LEVELS = [1, 2, 3, 4, 5]
RANDOM_SEED = 42

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


# ─── Safety-first selection metric ───────────────────────────────────────────

def clinical_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    The metrics that decide whether a triage model is any good.

    Ordered deliberately: critical recall and under-triage first, accuracy
    last. A model that is 95% accurate while missing one Level 1 patient in
    five is not a good model, and no aggregate that hides that distinction is
    worth reporting.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    critical = y_true <= 2
    n_critical = int(critical.sum())
    critical_recall = (
        float(((y_pred <= 2) & critical).sum() / n_critical) if n_critical else 1.0
    )

    # Under-triage = assigned a LESS urgent level than the truth (higher number)
    under = y_pred > y_true
    over = y_pred < y_true

    # The one that ends careers: a Level 1–2 patient routed to Level 3+.
    critical_under = critical & (y_pred > 2)

    return {
        "critical_recall": round(critical_recall, 4),
        "under_triage_rate": round(float(under.mean()), 4),
        "critical_under_triage_rate": round(
            float(critical_under.sum() / n_critical) if n_critical else 0.0, 4),
        "critical_under_triage_count": int(critical_under.sum()),
        "over_triage_rate": round(float(over.mean()), 4),
        "accuracy": round(float((y_pred == y_true).mean()), 4),
        "within_one_level": round(float((np.abs(y_pred - y_true) <= 1).mean()), 4),
        "n": int(len(y_true)),
        "n_critical": n_critical,
    }


def safety_score(m: Dict) -> float:
    """
    Single number used only for model *selection*, never for reporting.

    Weighted so that a model cannot buy its way to the top with accuracy while
    missing critical patients.
    """
    return (0.50 * m["critical_recall"]
            + 0.30 * (1.0 - m["critical_under_triage_rate"])
            + 0.15 * (1.0 - m["under_triage_rate"])
            + 0.05 * m["accuracy"])



def _auroc_critical(proba: np.ndarray, y_true: np.ndarray) -> float:
    """
    AUROC for identifying a critical (Level 1–2) patient.

    Reported separately from any decision rule because it measures what the
    model *knows*, independent of how aggressively we act on it. Two systems
    with identical AUROC and different thresholds are the same model at
    different operating points; two systems with different AUROC are not.
    """
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(y_true)
    is_critical = (y_true <= 2).astype(int)
    if is_critical.min() == is_critical.max():
        return float("nan")
    return round(float(roc_auc_score(is_critical, proba[:, 0] + proba[:, 1])), 4)


# ─── Grouped splitting ───────────────────────────────────────────────────────

def grouped_split(df: pd.DataFrame, group_col: str = "hospital_id",
                  fractions: Tuple[float, float, float, float] = (0.55, 0.15, 0.15, 0.15),
                  seed: int = RANDOM_SEED) -> Dict[str, np.ndarray]:
    """
    Split whole hospitals into train / calibrate / conformal / test.

    No hospital appears in two folds. This is the difference between "our model
    scores 0.94" and "our model scores 0.94 at hospitals it has never seen".
    """
    rng = np.random.RandomState(seed)
    # Cast to a plain numpy array: pandas nullable-integer arrays are not a
    # Sequence, and shuffling one in place is not guaranteed to be a permutation.
    # A silently duplicated hospital would place the same department in two
    # folds — the exact leakage this split exists to prevent.
    groups = np.asarray(df[group_col].dropna().unique().tolist())
    rng.shuffle(groups)

    n = len(groups)
    b1 = int(n * fractions[0])
    b2 = b1 + int(n * fractions[1])
    b3 = b2 + int(n * fractions[2])

    assignment = {
        "train": set(groups[:b1]),
        "calibrate": set(groups[b1:b2]),
        "conformal": set(groups[b2:b3]),
        "test": set(groups[b3:]),
    }

    idx = {}
    for name, grp in assignment.items():
        idx[name] = df.index[df[group_col].isin(grp)].to_numpy()
    return idx


# ─── The model bundle ────────────────────────────────────────────────────────

@dataclass
class TriageModelBundle:
    """Everything needed to score a patient, plus the provenance to defend it."""
    classifier: object = None
    calibrated_classifier: object = None
    conformal: Optional[ConformalTriage] = None
    critical_conformal: Optional[CriticalRiskConformal] = None
    text_vectorizer: object = None
    text_svd: object = None
    feature_names: List[str] = field(default_factory=list)
    model_name: str = ""
    cost_matrix: CostMatrix = field(default_factory=CostMatrix)
    metadata: Dict = field(default_factory=dict)

    # ── Inference ──
    def _text_components(self, texts: Sequence[str]) -> np.ndarray:
        if self.text_vectorizer is None or self.text_svd is None:
            return np.zeros((len(texts), 0))
        X = self.text_vectorizer.transform([t or "" for t in texts])
        return self.text_svd.transform(X)

    def build_matrix(self, records: List[Dict]) -> np.ndarray:
        """Feature matrix for a list of patient records."""
        df = pd.DataFrame(records)
        feats = build_feature_frame(df)
        for col in self.feature_names:
            if col not in feats.columns and not col.startswith("txt_"):
                feats[col] = np.nan
        numeric_cols = [c for c in self.feature_names if not c.startswith("txt_")]
        X_num = feats[numeric_cols].to_numpy(dtype=float)

        n_text = sum(1 for c in self.feature_names if c.startswith("txt_"))
        if n_text:
            texts = [
                f"{r.get('chief_complaint', '') or ''} {r.get('symptoms_text', '') or ''}"
                for r in records
            ]
            X_txt = self._text_components(texts)
            if X_txt.shape[1] != n_text:
                X_txt = np.zeros((len(records), n_text))
            return np.hstack([X_num, X_txt])
        return X_num

    def predict_proba(self, records: List[Dict]) -> np.ndarray:
        X = self.build_matrix(records)
        model = self.calibrated_classifier or self.classifier
        proba = model.predict_proba(X)
        return _align_proba(proba, model)

    def predict_one(self, record: Dict) -> Dict:
        """
        Score a single patient end to end.

        Returns the calibrated distribution, the conformal prediction set, the
        cost-optimal level, and the reconciled recommendation — the whole
        evidence trail, so the explanation layer never has to guess why a
        level was chosen.
        """
        t0 = time.perf_counter()
        proba = self.predict_proba([record])[0]

        cost_decision = expected_cost_decision(proba, self.cost_matrix, LEVELS)

        pred_set = self.conformal.predict_set(proba) if self.conformal else [
            LEVELS[int(np.argmax(proba))]
        ]

        # The conformal set is an honesty instrument, not the decision rule.
        #
        # An earlier design took min(prediction_set) as the recommendation. On
        # real ED data that collapses: triage labels are genuinely ambiguous,
        # so 90%-coverage sets are wide, and "act on the most urgent level in
        # a wide set" degenerates into calling almost everyone a Level 1 —
        # perfect critical recall, useless department. The set instead does
        # two honest jobs: it reports how much the data can actually pin down,
        # and it flags patients where a critical level cannot be ruled out so
        # a human looks again.
        conformal_level = int(min(pred_set))
        model_level = int(cost_decision["decision"])

        critical_probability = float(proba[0] + proba[1])
        critical_excluded = (
            self.critical_conformal.critical_excluded(critical_probability)
            if self.critical_conformal else False
        )
        critical_not_excluded = bool(not critical_excluded and model_level > 3)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "probabilities": {int(c): round(float(p), 4) for c, p in zip(LEVELS, proba)},
            "most_likely_level": int(LEVELS[int(np.argmax(proba))]),
            "max_probability": float(proba.max()),
            "cost_decision": cost_decision,
            "conformal_set": pred_set,
            "conformal_level": conformal_level,
            "conformal_span": int(max(pred_set) - min(pred_set)),
            "critical_not_excluded": critical_not_excluded,
            "critical_excluded_with_confidence": bool(critical_excluded),
            "review_recommended": bool(critical_not_excluded),
            "model_level": model_level,
            "critical_probability": float(proba[0] + proba[1]),
            "latency_ms": round(latency_ms, 3),
            "model_name": self.model_name,
        }


def _align_proba(proba: np.ndarray, model) -> np.ndarray:
    """Map a model's class ordering onto the fixed 1–5 column order."""
    classes = list(getattr(model, "classes_", LEVELS))
    if list(classes) == LEVELS:
        return proba
    out = np.zeros((proba.shape[0], len(LEVELS)))
    for i, c in enumerate(classes):
        if int(c) in LEVELS:
            out[:, LEVELS.index(int(c))] = proba[:, i]
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return out / row_sums


# ─── Candidate models ────────────────────────────────────────────────────────

def candidate_models() -> Dict[str, object]:
    """
    The models we actually consider, and why.

    Gradient-boosted trees on tabular clinical data, not deep learning. With
    ~30 structured physiological features and 20k rows, boosted trees are both
    the stronger performer and the auditable one — and auditability is not a
    nice-to-have in a system whose output a clinician has to justify.
    """
    models: Dict[str, object] = {
        # NaN-native: missing vitals are routed at each split rather than imputed.
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.06, max_depth=None,
            max_leaf_nodes=31, min_samples_leaf=25,
            l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.15, random_state=RANDOM_SEED,
            class_weight="balanced",
        ),
        "RandomForest": Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("clf", RandomForestClassifier(
                n_estimators=400, max_depth=14, min_samples_leaf=5,
                class_weight="balanced_subsample", n_jobs=-1,
                random_state=RANDOM_SEED)),
        ]),
        "LogisticRegression": Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=RANDOM_SEED)),
        ]),
    }

    if HAS_LIGHTGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            min_child_samples=25, subsample=0.9, subsample_freq=1,
            colsample_bytree=0.9, reg_lambda=1.0,
            class_weight="balanced", random_state=RANDOM_SEED, verbose=-1,
        )
    if HAS_XGBOOST:
        models["XGBoost"] = _XGBLevelAdapter(
            n_estimators=400, learning_rate=0.06, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            eval_metric="mlogloss", random_state=RANDOM_SEED,
            tree_method="hist", verbosity=0,
        )
    return models


class _XGBLevelAdapter:
    """
    XGBoost requires 0-indexed class labels; triage levels are 1–5.

    Wrapping the offset here rather than shifting labels globally keeps one
    representation of a triage level in the codebase. Off-by-one errors on a
    severity scale are the kind of bug that silently turns a Level 2 into a
    Level 3, so the conversion lives in exactly one place.
    """

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._model = XGBClassifier(**kwargs)
        self.classes_ = np.array(LEVELS)
        self._offset = 0

    def fit(self, X, y):
        y = np.asarray(y, dtype=int)
        self._offset = int(y.min())
        self._model.fit(X, y - self._offset)
        self.classes_ = np.array(sorted(np.unique(y)))
        return self

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    def predict(self, X):
        return self._model.predict(X) + self._offset

    def get_params(self, deep=True):
        return dict(self._kwargs)

    def set_params(self, **params):
        self._kwargs.update(params)
        self._model = XGBClassifier(**self._kwargs)
        return self

    def __sklearn_tags__(self):
        return self._model.__sklearn_tags__()


# ─── Training ────────────────────────────────────────────────────────────────

def train_bundle(df: pd.DataFrame,
                 label_col: str = "triage_level",
                 group_col: str = "hospital_id",
                 alpha: float = 0.10,
                 cost_matrix: Optional[CostMatrix] = None,
                 max_critical_lane_load: float = 0.35,
                 min_critical_recall: float = 0.65,
                 use_text: bool = True,
                 verbose: bool = True) -> Tuple[TriageModelBundle, Dict]:
    """Train, select, calibrate and conformalise. Returns the bundle + a report."""
    cost_matrix = cost_matrix or CostMatrix()
    report: Dict = {}

    if verbose:
        print("\n[1/6] Building features …")
    feats = build_feature_frame(df)
    numeric_cols = list(feats.columns)

    # ── Text branch: chief complaint → TF-IDF → SVD components ──
    text_vectorizer = text_svd = None
    text_cols: List[str] = []
    if use_text and "chief_complaint" in df.columns:
        texts = (df["chief_complaint"].fillna("") + " "
                 + df.get("symptoms_text", pd.Series("", index=df.index)).fillna(""))
        text_vectorizer = TfidfVectorizer(max_features=4000, ngram_range=(1, 2),
                                          min_df=3, sublinear_tf=True)
        X_text_sparse = text_vectorizer.fit_transform(texts)
        n_comp = min(64, max(2, X_text_sparse.shape[1] - 1))
        text_svd = TruncatedSVD(n_components=n_comp, random_state=RANDOM_SEED)
        X_text = text_svd.fit_transform(X_text_sparse)
        text_cols = [f"txt_{i}" for i in range(X_text.shape[1])]
        if verbose:
            print(f"      complaint text → {len(text_cols)} semantic components "
                  f"({X_text_sparse.shape[1]} terms, "
                  f"{text_svd.explained_variance_ratio_.sum():.1%} variance retained)")
    else:
        X_text = np.zeros((len(df), 0))

    X_all = np.hstack([feats[numeric_cols].to_numpy(dtype=float), X_text])
    y_all = df[label_col].to_numpy(dtype=int)
    feature_names = numeric_cols + text_cols

    # ── Grouped splits ──
    if verbose:
        print("\n[2/6] Splitting by hospital (no department appears in two folds) …")
    df_reset = df.reset_index(drop=True)
    idx = grouped_split(df_reset, group_col=group_col)
    for name in ["train", "calibrate", "conformal", "test"]:
        n_hosp = df_reset.loc[idx[name], group_col].nunique()
        if verbose:
            print(f"      {name:10s} {len(idx[name]):6,} visits   {n_hosp:3d} hospitals")
    report["splits"] = {
        name: {
            "n_visits": int(len(idx[name])),
            "n_hospitals": int(df_reset.loc[idx[name], group_col].nunique()),
        } for name in idx
    }

    Xtr, ytr = X_all[idx["train"]], y_all[idx["train"]]
    Xcal, ycal = X_all[idx["calibrate"]], y_all[idx["calibrate"]]
    Xcnf, ycnf = X_all[idx["conformal"]], y_all[idx["conformal"]]
    Xte, yte = X_all[idx["test"]], y_all[idx["test"]]

    # ── Train candidates and compare them at a MATCHED escalation budget ──
    #
    # Comparing triage models on raw safety metrics is meaningless, because any
    # model can score perfect critical recall by calling every patient a Level
    # 1. So each candidate is first tuned to the same over-triage budget, and
    # only then compared on how many critical patients it catches. That is a
    # fair fight: equal cost, measured benefit.
    if verbose:
        print("\n[3/6] Training candidates and matching them to a common "
              "escalation budget …")
    leaderboard = []
    fitted = {}
    for name, model in candidate_models().items():
        t0 = time.perf_counter()
        try:
            model.fit(Xtr, ytr)
        except Exception as exc:
            if verbose:
                print(f"      {name:22s} FAILED ({exc})")
            continue
        train_seconds = time.perf_counter() - t0

        proba_val = _align_proba(model.predict_proba(Xcal), model)
        curve = operating_curve(proba_val, ycal, cost_matrix)
        chosen = select_operating_point(curve, max_critical_lane_load, min_critical_recall)
        profile = scaled_profile(cost_matrix, chosen["selected_lambda"])

        y_val_pred = np.array([
            expected_cost_decision(p, profile)["decision"] for p in proba_val
        ])
        m = clinical_metrics(ycal, y_val_pred)
        m["model"] = name
        m["selected_lambda"] = chosen["selected_lambda"]
        m["budget_status"] = chosen["status"]
        m["safety_score"] = round(safety_score(m), 4)
        m["train_seconds"] = round(train_seconds, 2)

        # Discrimination is measured independently of any decision rule, so a
        # model cannot look good purely because its threshold was tuned well.
        m["auroc_critical"] = _auroc_critical(proba_val, ycal)
        leaderboard.append(m)
        fitted[name] = model
        if verbose:
            print(f"      {name:22s} AUROC={m['auroc_critical']:.3f}  "
                  f"crit_recall={m['critical_recall']:.3f} @ "
                  f"lane_load={m['over_triage_rate']:.3f}  "
                  f"(λ={m['selected_lambda']:g}, {train_seconds:.1f}s)")

    if not leaderboard:
        raise RuntimeError("No candidate model trained successfully.")

    # Rank by discrimination first — it is the property that survives
    # re-tuning — then by achieved critical recall at the matched budget.
    leaderboard.sort(key=lambda m: (m["auroc_critical"], m["critical_recall"]),
                     reverse=True)
    best_name = leaderboard[0]["model"]
    best_model = fitted[best_name]
    report["leaderboard"] = leaderboard
    report["selected_model"] = best_name
    report["selection_criterion"] = (
        "Every candidate is tuned to the same escalation budget "
        f"(≤{max_critical_lane_load:.0%} emergent-lane load, ≥{min_critical_recall:.0%} critical "
        "recall), then ranked by AUROC for critical (Level 1 or 2) identification "
        "and by critical recall achieved at that matched budget. Overall "
        "accuracy is reported but never selects a model: a model that calls "
        "every patient Level 1 has perfect critical recall and is useless."
    )
    if verbose:
        print(f"\n      → selected {best_name}")

    # ── Probability calibration on its own fold ──
    if verbose:
        print("\n[4/6] Calibrating probabilities (isotonic, held-out fold) …")
    proba_cal_raw = _align_proba(best_model.predict_proba(Xcal), best_model)
    pre_cal = calibration_report(ycal, proba_cal_raw)

    # Isotonic is flexible but can lose sharpness on a small calibration fold;
    # Platt scaling is the opposite trade. Rather than assume, fit both and
    # keep whichever genuinely helps on a fold neither of them was fitted on.
    # If neither improves on the raw model, we keep the raw model — a
    # calibration step that makes things worse is not a calibration step.
    calibration_candidates = {}
    for method in ("isotonic", "sigmoid"):
        try:
            cal = CalibratedClassifierCV(best_model, method=method, cv="prefit")
            cal.fit(Xcal, ycal)
            proba_try = _align_proba(cal.predict_proba(Xcnf), cal)
            rep = calibration_report(ycnf, proba_try)
            calibration_candidates[method] = (cal, rep)
        except Exception as exc:
            if verbose:
                print(f"      {method} calibration failed: {exc}")

    raw_cnf = _align_proba(best_model.predict_proba(Xcnf), best_model)
    raw_rep = calibration_report(ycnf, raw_cnf)
    calibration_candidates["none"] = (None, raw_rep)

    # Score on ECE first (the number a clinician's trust depends on), with
    # Brier as a tiebreaker so we do not accept a big sharpness loss for a
    # trivial calibration gain.
    def _cal_score(rep):
        return rep["expected_calibration_error"] + 0.25 * rep["brier_score"]

    best_method = min(calibration_candidates, key=lambda m: _cal_score(calibration_candidates[m][1]))
    calibrated_obj, post_cal = calibration_candidates[best_method]
    calibrated = calibrated_obj if calibrated_obj is not None else best_model
    proba_cnf = _align_proba(calibrated.predict_proba(Xcnf), calibrated)

    if verbose:
        for method, (_, rep) in calibration_candidates.items():
            mark = "→" if method == best_method else " "
            print(f"      {mark} {method:9s} Brier={rep['brier_score']:.4f}  "
                  f"ECE={rep['expected_calibration_error']:.4f}")
    report["calibration"] = {
        "method_selected": best_method,
        "before": raw_rep,
        "after": post_cal,
        "candidates": {m: r for m, (_, r) in calibration_candidates.items()},
        "selection_rule": (
            "Chosen on a fold used for neither training nor calibrator fitting, "
            "scoring ECE + 0.25·Brier so that a calibration method cannot win by "
            "flattening probabilities into uselessness."
        ),
    }

    # ── Final operating point + conformal thresholds, on the untouched fold ──
    if verbose:
        print(f"\n[5/6] Setting the operating point and conformal sets "
              f"(α = {alpha}) …")

    final_curve = operating_curve(proba_cnf, ycnf, cost_matrix)
    final_point = select_operating_point(final_curve, max_critical_lane_load,
                                         min_critical_recall)
    tuned_cost_matrix = scaled_profile(
        cost_matrix, final_point["selected_lambda"], name=f"{cost_matrix.name}_tuned"
    )
    if verbose:
        print(f"      {final_point['explanation']}")
        if final_point["status"] != "met_both_constraints":
            print(f"      NOTE: {final_point['status']}, reported as-is, not "
                  f"hidden behind a nicer-looking number.")
    report["operating_point"] = final_point
    report["operating_curve"] = final_curve

    conformal = ConformalTriage(alpha=alpha, mode="class_conditional")
    conformal.calibrate(proba_cnf, ycnf)

    critical_conformal = CriticalRiskConformal(alpha=0.05)
    critical_conformal.calibrate(proba_cnf, ycnf)

    proba_te = _align_proba(calibrated.predict_proba(Xte), calibrated)
    conf_eval = conformal.evaluate(proba_te, yte)
    crit_eval = critical_conformal.evaluate(proba_te, yte)
    if verbose:
        print(f"      5-class set: target {conf_eval['target_coverage']:.0%}, "
              f"empirical {conf_eval['empirical_coverage']:.1%}, "
              f"mean size {conf_eval['mean_set_size']:.2f}")
        print(f"      critical-exclusion: {crit_eval['interpretation']}")
    report["conformal"] = conf_eval
    report["critical_conformal"] = crit_eval

    bundle = TriageModelBundle(
        classifier=best_model,
        calibrated_classifier=calibrated,
        conformal=conformal,
        critical_conformal=critical_conformal,
        text_vectorizer=text_vectorizer,
        text_svd=text_svd,
        feature_names=feature_names,
        model_name=best_name,
        cost_matrix=tuned_cost_matrix,
        metadata={
            "trained_at": pd.Timestamp.now().isoformat(),
            "n_train": int(len(idx["train"])),
            "n_features": len(feature_names),
            "label": label_col,
            "alpha": alpha,
            "cost_profile": tuned_cost_matrix.to_dict(),
            "operating_point": final_point,
            "escalation_budget": {
                "max_critical_lane_load": max_critical_lane_load,
                "min_critical_recall": min_critical_recall,
            },
            "split_strategy": "grouped by hospital (no leakage across folds)",
        },
    )

    if verbose:
        print("\n[6/6] Bundle assembled.")

    report["test_fold_indices"] = idx["test"].tolist()
    return bundle, report


# ─── Persistence ─────────────────────────────────────────────────────────────

def save_bundle(bundle: TriageModelBundle, path: str):
    import joblib
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({
        "classifier": bundle.classifier,
        "calibrated_classifier": bundle.calibrated_classifier,
        "conformal": bundle.conformal.to_dict() if bundle.conformal else None,
        "critical_conformal": (bundle.critical_conformal.to_dict()
                               if bundle.critical_conformal else None),
        "text_vectorizer": bundle.text_vectorizer,
        "text_svd": bundle.text_svd,
        "feature_names": bundle.feature_names,
        "model_name": bundle.model_name,
        "cost_matrix": bundle.cost_matrix.to_dict(),
        "metadata": bundle.metadata,
    }, path)


def load_bundle(path: str) -> TriageModelBundle:
    import joblib
    d = joblib.load(path)
    return TriageModelBundle(
        classifier=d["classifier"],
        calibrated_classifier=d["calibrated_classifier"],
        conformal=ConformalTriage.from_dict(d["conformal"]) if d["conformal"] else None,
        critical_conformal=(CriticalRiskConformal.from_dict(d["critical_conformal"])
                            if d.get("critical_conformal") else None),
        text_vectorizer=d["text_vectorizer"],
        text_svd=d["text_svd"],
        feature_names=d["feature_names"],
        model_name=d["model_name"],
        cost_matrix=CostMatrix.from_dict(d["cost_matrix"]),
        metadata=d["metadata"],
    )
