"""
PulseGuard — Calibration & Conformal Uncertainty
======================================================

A triage assistant that says "Level 3, 99% confident" on every patient is
worse than one that says nothing, because it teaches staff to stop reading the
number. Two things have to be true before a confidence figure deserves screen
space:

1. **It has to be calibrated.** When the system says 80%, it should be right
   about 80% of the time. We measure this (Brier score, expected calibration
   error, reliability curves) and we fix it (isotonic regression fitted on a
   held-out calibration split, never on training data).

2. **It has to come with a guarantee.** Calibration is an average property; a
   nurse cares about *this* patient. Split-conformal prediction converts the
   model's scores into a prediction *set* with a finite-sample coverage
   guarantee: at α = 0.10 the true triage level is inside the set for at least
   90% of patients, and that holds without assuming the model is correct or
   the data is Gaussian — only that calibration and test data are
   exchangeable.

The five-class set is an honesty instrument: it reports how much the data can
actually pin down, and triage data on real patients frequently cannot pin down
much. The guarantee that drives *action* is the binary one below
(`CriticalRiskConformal`) — "can a critical presentation be excluded?" —
because collapsing to the question with the highest stakes concentrates all
the statistical power there, and yields a set a nurse can act on rather than a
four-level range they cannot.

References:
  - Vovk et al., Algorithmic Learning in a Random World (conformal prediction)
  - Sadinle, Lei & Wasserman (2019), least-ambiguous set-valued classifiers
  - Guo et al. (2017), On Calibration of Modern Neural Networks (ECE)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

LEVELS = [1, 2, 3, 4, 5]


# ─── Calibration metrics ─────────────────────────────────────────────────────

def brier_score_multiclass(y_true: np.ndarray, proba: np.ndarray,
                           classes: Sequence[int] = LEVELS) -> float:
    """
    Multiclass Brier score (mean squared error of the probability vector).

    Lower is better; 0 is perfect. Unlike accuracy it punishes a model for
    being confidently wrong, which is the failure mode that matters here.
    """
    y_true = np.asarray(y_true)
    onehot = np.zeros_like(proba, dtype=float)
    class_index = {c: i for i, c in enumerate(classes)}
    for row, y in enumerate(y_true):
        if y in class_index:
            onehot[row, class_index[y]] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray,
                               classes: Sequence[int] = LEVELS,
                               n_bins: int = 10) -> Dict:
    """
    Expected and maximum calibration error over confidence bins.

    ECE is the average gap between "how confident the model was" and "how often
    it was right", weighted by how many patients fall in each confidence bin.
    """
    y_true = np.asarray(y_true)
    classes = np.asarray(classes)
    confidence = proba.max(axis=1)
    predicted = classes[proba.argmax(axis=1)]
    correct = (predicted == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, mce, total = 0.0, 0.0, len(y_true)
    curve = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidence > lo) & (confidence <= hi) if i > 0 else (confidence >= lo) & (confidence <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        avg_conf = float(confidence[mask].mean())
        avg_acc = float(correct[mask].mean())
        gap = abs(avg_conf - avg_acc)
        ece += (n / total) * gap
        mce = max(mce, gap)
        curve.append({
            "bin_lower": round(float(lo), 3),
            "bin_upper": round(float(hi), 3),
            "n": n,
            "mean_confidence": round(avg_conf, 4),
            "accuracy": round(avg_acc, 4),
            "gap": round(gap, 4),
        })

    return {
        "ece": round(float(ece), 4),
        "mce": round(float(mce), 4),
        "n_bins": n_bins,
        "reliability_curve": curve,
    }


def calibration_report(y_true: np.ndarray, proba: np.ndarray,
                       classes: Sequence[int] = LEVELS) -> Dict:
    """Everything we report about how trustworthy the confidence number is."""
    ece = expected_calibration_error(y_true, proba, classes)
    return {
        "brier_score": round(brier_score_multiclass(y_true, proba, classes), 4),
        "expected_calibration_error": ece["ece"],
        "max_calibration_error": ece["mce"],
        "reliability_curve": ece["reliability_curve"],
        "mean_confidence": round(float(proba.max(axis=1).mean()), 4),
        "accuracy": round(float((np.asarray(classes)[proba.argmax(axis=1)] == np.asarray(y_true)).mean()), 4),
        "interpretation": (
            "Calibration gap (mean confidence − accuracy) of "
            f"{round(float(proba.max(axis=1).mean()) - float((np.asarray(classes)[proba.argmax(axis=1)] == np.asarray(y_true)).mean()), 4):+.4f}. "
            "Positive means the system is overconfident."
        ),
    }


# ─── Split-conformal prediction ──────────────────────────────────────────────

class ConformalTriage:
    """
    Split-conformal prediction sets for triage levels.

    Two modes:

    * **marginal** — one threshold for all patients. Guarantees overall
      coverage of at least 1 − α.
    * **class-conditional (Mondrian)** — a separate threshold per true level.
      Guarantees coverage *within each severity level*, which is what safety
      actually requires: a marginal guarantee can be satisfied while
      systematically failing the 1.8% of patients who are Level 1, because
      they barely move the average.

    We default to class-conditional for exactly that reason.
    """

    def __init__(self, alpha: float = 0.10, mode: str = "class_conditional",
                 classes: Sequence[int] = LEVELS):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self.mode = mode
        self.classes = list(classes)
        self.q_hat: Optional[float] = None
        self.q_hat_by_class: Dict[int, float] = {}
        self.n_calibration = 0
        self.is_fitted = False

    def _scores(self, proba: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Non-conformity score: 1 − p(true label). Higher = worse surprise."""
        idx = {c: i for i, c in enumerate(self.classes)}
        return np.array([
            1.0 - proba[i, idx[y]] if y in idx else 1.0
            for i, y in enumerate(y_true)
        ])

    def calibrate(self, proba_cal: np.ndarray, y_cal: np.ndarray) -> "ConformalTriage":
        """
        Fit the conformal thresholds on a calibration split.

        This split must be disjoint from both training and test data — that
        disjointness is the entire source of the coverage guarantee.
        """
        y_cal = np.asarray(y_cal)
        scores = self._scores(proba_cal, y_cal)
        self.n_calibration = len(scores)

        # Finite-sample corrected quantile
        n = len(scores)
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q_hat = float(np.quantile(scores, level, method="higher"))

        for c in self.classes:
            mask = y_cal == c
            n_c = int(mask.sum())
            if n_c < 10:
                # Too few calibration points for a per-class guarantee; fall
                # back to the marginal threshold rather than invent one from
                # three patients.
                self.q_hat_by_class[c] = self.q_hat
                continue
            level_c = min(1.0, np.ceil((n_c + 1) * (1 - self.alpha)) / n_c)
            self.q_hat_by_class[c] = float(
                np.quantile(scores[mask], level_c, method="higher")
            )

        self.is_fitted = True
        return self

    def predict_set(self, proba_row: np.ndarray) -> List[int]:
        """Return the prediction set for one patient's probability vector."""
        if not self.is_fitted:
            raise RuntimeError("ConformalTriage must be calibrated before use.")

        pred_set = []
        for i, c in enumerate(self.classes):
            threshold = (self.q_hat_by_class.get(c, self.q_hat)
                         if self.mode == "class_conditional" else self.q_hat)
            if (1.0 - proba_row[i]) <= threshold:
                pred_set.append(c)

        # A guarantee is worthless if the set can be empty — an empty set gives
        # a nurse nothing to act on. Fall back to the single most likely level.
        if not pred_set:
            pred_set = [self.classes[int(np.argmax(proba_row))]]

        # Triage levels are ORDINAL, so the set has to be an interval.
        #
        # Raw per-class thresholding can return {1, 2, 4, 5}: "this patient is
        # critical, or nearly non-urgent, but definitely not in between."
        # That is a statement no clinician can act on and no clinician will
        # trust. Taking the convex hull yields a contiguous severity range,
        # and since the hull is a superset of the original set, the coverage
        # guarantee is preserved — the set can only get safer, never less
        # likely to contain the truth.
        lo, hi = min(pred_set), max(pred_set)
        return [c for c in self.classes if lo <= c <= hi]

    def predict_sets(self, proba: np.ndarray) -> List[List[int]]:
        return [self.predict_set(row) for row in proba]

    def evaluate(self, proba: np.ndarray, y_true: np.ndarray) -> Dict:
        """Empirical coverage and set size — does the guarantee actually hold?"""
        y_true = np.asarray(y_true)
        sets = self.predict_sets(proba)
        covered = np.array([y in s for y, s in zip(y_true, sets)])
        sizes = np.array([len(s) for s in sets])

        by_class = {}
        for c in self.classes:
            mask = y_true == c
            if mask.sum() == 0:
                continue
            by_class[int(c)] = {
                "n": int(mask.sum()),
                "coverage": round(float(covered[mask].mean()), 4),
                "mean_set_size": round(float(sizes[mask].mean()), 3),
            }

        return {
            "alpha": self.alpha,
            "target_coverage": round(1 - self.alpha, 3),
            "empirical_coverage": round(float(covered.mean()), 4),
            "mean_set_size": round(float(sizes.mean()), 3),
            "pct_singleton_sets": round(float((sizes == 1).mean()), 4),
            "coverage_by_true_level": by_class,
            "mode": self.mode,
            "n_calibration": self.n_calibration,
            "guarantee_holds": bool(covered.mean() >= (1 - self.alpha) - 0.02),
        }

    def to_dict(self) -> Dict:
        return {
            "alpha": self.alpha,
            "mode": self.mode,
            "classes": self.classes,
            "q_hat": self.q_hat,
            "q_hat_by_class": {str(k): v for k, v in self.q_hat_by_class.items()},
            "n_calibration": self.n_calibration,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ConformalTriage":
        obj = cls(alpha=d["alpha"], mode=d["mode"], classes=d["classes"])
        obj.q_hat = d["q_hat"]
        obj.q_hat_by_class = {int(k): v for k, v in d["q_hat_by_class"].items()}
        obj.n_calibration = d["n_calibration"]
        obj.is_fitted = True
        return obj


class CriticalRiskConformal:
    """
    Binary conformal predictor for the only question that has to be answered
    with a guarantee: **can a critical presentation be excluded?**

    The five-class predictor is honest but blunt — on genuinely ambiguous
    triage data its 90%-coverage sets span three or four levels, which tells a
    nurse little. Collapsing to the binary question concentrates all the
    statistical power where the clinical stakes are, and yields a usable
    decision: either a Level 1–2 presentation is ruled out at the stated
    confidence, or it is not and a human looks again.

    This is the predictor the CRITICAL_NOT_EXCLUDED safety rule consumes.
    """

    def __init__(self, alpha: float = 0.05):
        # A tighter α than the five-class predictor: this guards the failure
        # mode with the highest cost, so we buy more coverage for it.
        self.alpha = alpha
        self.threshold: Optional[float] = None
        self.n_calibration = 0
        self.is_fitted = False

    def calibrate(self, proba_cal: np.ndarray, y_cal: np.ndarray) -> "CriticalRiskConformal":
        """
        Find the P(critical) threshold below which critical can be excluded
        with 1 − α confidence.

        Calibrated only on patients who truly were critical: the guarantee we
        want is "of all genuinely critical patients, at most α% fall below this
        threshold", which is the false-negative rate that matters.
        """
        y_cal = np.asarray(y_cal)
        critical_scores = (proba_cal[:, 0] + proba_cal[:, 1])[y_cal <= 2]
        self.n_calibration = int(len(critical_scores))

        if self.n_calibration < 20:
            self.threshold = 0.0     # too few points to justify any exclusion
        else:
            n = self.n_calibration
            level = max(0.0, np.floor((n + 1) * self.alpha) / n)
            self.threshold = float(np.quantile(critical_scores, level, method="lower"))

        self.is_fitted = True
        return self

    def critical_excluded(self, critical_probability: float) -> bool:
        if not self.is_fitted or self.threshold is None:
            return False
        return bool(critical_probability < self.threshold)

    def evaluate(self, proba: np.ndarray, y_true: np.ndarray) -> Dict:
        y_true = np.asarray(y_true)
        scores = proba[:, 0] + proba[:, 1]
        excluded = np.array([self.critical_excluded(s) for s in scores])
        truly_critical = y_true <= 2

        n_crit = int(truly_critical.sum())
        missed = int((excluded & truly_critical).sum())

        return {
            "alpha": self.alpha,
            "threshold": round(float(self.threshold or 0.0), 4),
            "target_max_miss_rate": self.alpha,
            "empirical_miss_rate": round(missed / n_crit, 4) if n_crit else 0.0,
            "n_critical": n_crit,
            "n_critical_missed": missed,
            "pct_patients_cleared": round(float(excluded.mean()), 4),
            "guarantee_holds": bool((missed / n_crit if n_crit else 0.0)
                                    <= self.alpha + 0.02),
            "n_calibration": self.n_calibration,
            "interpretation": (
                f"{float(excluded.mean()):.1%} of patients can have a critical "
                f"presentation excluded with {1 - self.alpha:.0%} confidence; "
                f"of genuinely critical patients, "
                f"{(missed / n_crit if n_crit else 0):.1%} fall below the "
                f"threshold (target ≤{self.alpha:.0%})."
            ),
        }

    def to_dict(self) -> Dict:
        return {"alpha": self.alpha, "threshold": self.threshold,
                "n_calibration": self.n_calibration}

    @classmethod
    def from_dict(cls, d: Dict) -> "CriticalRiskConformal":
        obj = cls(alpha=d["alpha"])
        obj.threshold = d["threshold"]
        obj.n_calibration = d["n_calibration"]
        obj.is_fitted = True
        return obj


def uncertainty_band_from_set(pred_set: List[int]) -> str:
    """
    Translate a conformal set into the three-band vocabulary the UI speaks.

    The band is derived from the width of the guaranteed set, so "high
    uncertainty" now has a precise meaning — the data genuinely cannot
    distinguish between three or more severity levels at 90% coverage — rather
    than being a threshold someone picked.
    """
    span = max(pred_set) - min(pred_set) if pred_set else 4
    if len(pred_set) == 1:
        return "low"
    if span <= 1:
        return "moderate"
    return "high"
