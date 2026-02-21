"""Component-outcome analysis for transfer learning models.

Evaluates each model separately on ED-only and hospitalisation-only outcomes
to ensure the composite acute-care utilisation result is not driven entirely
by one component.  Mirrors the primary evaluation but uses component-specific
binary labels derived from the raw Virginia data.

The raw Medicaid data must contain two columns alongside the composite outcome:
    ed_utilization_12m      — 1 if ≥1 ED visit in 12-month outcome window
    hosp_utilization_12m    — 1 if ≥1 inpatient hospitalisation in 12-month window

If these columns are absent the module logs a warning and returns empty frames
so the main pipeline degrades gracefully.

Reference:
    Basu S et al. Transferring Healthcare Risk Prediction Models Between Two
    Medicaid Populations.  npj Health Systems (under review).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from evaluation_metrics import compute_classification_metrics

logger = logging.getLogger(__name__)

# Column names expected in the raw (pre-encoding) target data frame.
ED_OUTCOME_COL = "ed_utilization_12m"
HOSP_OUTCOME_COL = "hosp_utilization_12m"

# Labels used in output tables.
COMPONENT_LABELS: Dict[str, str] = {
    ED_OUTCOME_COL: "ED-only",
    HOSP_OUTCOME_COL: "Hospitalisation-only",
}


@dataclass
class ComponentOutcomeBundle:
    """Holds per-component labels aligned to the test indices."""

    ed_y_test: Optional[np.ndarray]       # 1/0 ED outcome, test set
    hosp_y_test: Optional[np.ndarray]     # 1/0 hosp outcome, test set
    ed_available: bool
    hosp_available: bool


def extract_component_labels(
    raw_target_df: pd.DataFrame,
    test_indices: np.ndarray,
) -> ComponentOutcomeBundle:
    """Extract component outcome labels for the test split.

    Parameters
    ----------
    raw_target_df : DataFrame
        Post-cohort-criteria, pre-encoding Virginia data.  Must be row-aligned
        to the same order used when building ``target_bundle.X_test``
        (i.e. after ``reset_index(drop=True)`` and temporal sort).
    test_indices : array of int
        Row indices (into ``raw_target_df``) corresponding to the test split.

    Returns
    -------
    ComponentOutcomeBundle
    """
    ed_avail = ED_OUTCOME_COL in raw_target_df.columns
    hosp_avail = HOSP_OUTCOME_COL in raw_target_df.columns

    if not ed_avail:
        logger.warning(
            "Column '%s' not found in target data; ED-only analysis unavailable.", ED_OUTCOME_COL
        )
    if not hosp_avail:
        logger.warning(
            "Column '%s' not found in target data; hospitalisation-only analysis unavailable.",
            HOSP_OUTCOME_COL,
        )

    ed_y = raw_target_df[ED_OUTCOME_COL].iloc[test_indices].to_numpy(dtype=int) if ed_avail else None
    hosp_y = raw_target_df[HOSP_OUTCOME_COL].iloc[test_indices].to_numpy(dtype=int) if hosp_avail else None

    return ComponentOutcomeBundle(
        ed_y_test=ed_y,
        hosp_y_test=hosp_y,
        ed_available=ed_avail,
        hosp_available=hosp_avail,
    )


def _bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 789,
) -> Tuple[float, float]:
    """Return (lower, upper) bootstrap CI for AUC."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs: List[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    if len(aucs) < 10:
        return float("nan"), float("nan")
    lo = float(np.percentile(aucs, 100 * alpha / 2))
    hi = float(np.percentile(aucs, 100 * (1 - alpha / 2)))
    return lo, hi


def evaluate_component_outcomes(
    trained_models: Dict[str, Any],
    component_bundle: ComponentOutcomeBundle,
    target_X_test: np.ndarray,
    n_bootstrap: int = 1000,
) -> Dict[str, pd.DataFrame]:
    """Evaluate all models on ED-only and hospitalisation-only outcomes.

    Parameters
    ----------
    trained_models : dict of model_name -> fitted model
        Must support ``predict_proba``.
    component_bundle : ComponentOutcomeBundle
        Aligned component labels for the test set.
    target_X_test : array of shape (n_test, n_features)
        Feature matrix for the test set (same ordering as component labels).
    n_bootstrap : int
        Number of bootstrap iterations for CI estimation.

    Returns
    -------
    dict with keys ``'ed_comparison'`` and ``'hosp_comparison'``, each a
    DataFrame mirroring the structure of the primary model_comparison table.
    """
    results: Dict[str, pd.DataFrame] = {}

    component_map = []
    if component_bundle.ed_available and component_bundle.ed_y_test is not None:
        component_map.append(("ed_comparison", COMPONENT_LABELS[ED_OUTCOME_COL], component_bundle.ed_y_test))
    if component_bundle.hosp_available and component_bundle.hosp_y_test is not None:
        component_map.append(("hosp_comparison", COMPONENT_LABELS[HOSP_OUTCOME_COL], component_bundle.hosp_y_test))

    for result_key, outcome_label, y_component in component_map:
        rows: List[Dict[str, Any]] = []
        for model_name, model in trained_models.items():
            try:
                y_score = model.predict_proba(target_X_test)[:, 1]
            except Exception as exc:
                logger.error(
                    "Model %s failed predict_proba during component analysis: %s", model_name, exc
                )
                continue

            if len(np.unique(y_component)) < 2:
                logger.warning(
                    "Component outcome '%s' has fewer than 2 classes in test set; skipping.", outcome_label
                )
                continue

            metrics, threshold = compute_classification_metrics(y_component, y_score)
            auc_lo, auc_hi = _bootstrap_auc_ci(y_component, y_score, n_bootstrap=n_bootstrap)

            rows.append(
                {
                    "Model": model_name,
                    "Outcome": outcome_label,
                    "N_test": len(y_component),
                    "EventRate": round(float(np.mean(y_component)), 4),
                    "AUC": round(metrics["auc"], 4),
                    "AUC_CI_lower": round(auc_lo, 4) if not np.isnan(auc_lo) else np.nan,
                    "AUC_CI_upper": round(auc_hi, 4) if not np.isnan(auc_hi) else np.nan,
                    "Sensitivity": round(metrics["sensitivity"], 4),
                    "Specificity": round(metrics["specificity"], 4),
                    "YoudensJ": round(metrics["youdens_j"], 4) if not np.isnan(metrics["youdens_j"]) else np.nan,
                    "Precision": round(metrics["precision"], 4),
                    "BrierScore": round(metrics["brier_score"], 4),
                    "OptimalThreshold": round(threshold, 4),
                }
            )

        if rows:
            df = pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)
        else:
            df = pd.DataFrame()
        results[result_key] = df
        logger.info(
            "Component outcome '%s': evaluated %d models.", outcome_label, len(rows)
        )

    return results


class ComponentOutcomeAnalyzer:
    """Orchestrate component-outcome analysis across all models."""

    def __init__(self, config: Dict[str, Any], n_bootstrap: int = 1000):
        self.config = config
        self.n_bootstrap = n_bootstrap

    def analyze(
        self,
        trained_models: Dict[str, Any],
        raw_target_df: pd.DataFrame,
        test_indices: np.ndarray,
        target_X_test: np.ndarray,
    ) -> Dict[str, pd.DataFrame]:
        """
        Run component outcome analysis.

        Parameters
        ----------
        trained_models : dict
            Fitted model objects with ``predict_proba``.
        raw_target_df : DataFrame
            Post-cohort Virginia data (pre-encoding), row-aligned.
        test_indices : array of int
            Indices into ``raw_target_df`` for the test split.
        target_X_test : array of shape (n_test, n_features)

        Returns
        -------
        dict with 'ed_comparison' and/or 'hosp_comparison' DataFrames,
        or empty DataFrames when component columns are not available.
        """
        if not trained_models:
            logger.warning("No trained models supplied; skipping component outcome analysis.")
            return {"ed_comparison": pd.DataFrame(), "hosp_comparison": pd.DataFrame()}

        component_bundle = extract_component_labels(raw_target_df, test_indices)

        if not component_bundle.ed_available and not component_bundle.hosp_available:
            logger.warning(
                "Neither '%s' nor '%s' found in target data; component analysis skipped.",
                ED_OUTCOME_COL,
                HOSP_OUTCOME_COL,
            )
            return {"ed_comparison": pd.DataFrame(), "hosp_comparison": pd.DataFrame()}

        return evaluate_component_outcomes(
            trained_models=trained_models,
            component_bundle=component_bundle,
            target_X_test=target_X_test,
            n_bootstrap=self.n_bootstrap,
        )
