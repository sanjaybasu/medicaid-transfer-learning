"""Calibration assessment and post-hoc correction utilities."""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from evaluation_metrics import compute_classification_metrics

logger = logging.getLogger(__name__)


def hosmer_lemeshow_test(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    data = pd.DataFrame({"y": y_true, "p": y_prob})
    data.sort_values("p", inplace=True)
    data["bin"] = pd.qcut(data["p"], q=n_bins, duplicates="drop")

    grouped = data.groupby("bin", observed=False)
    observed = grouped["y"].sum()
    expected = grouped["p"].sum()
    total = grouped.size()

    with np.errstate(divide="ignore", invalid="ignore"):
        hl_stat = np.nansum((observed - expected) ** 2 / (expected * (1 - expected / total)))
    df = max(len(grouped) - 2, 1)
    p_value = 1 - chi2.cdf(hl_stat, df)
    return float(p_value)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    digitized = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for bin_idx in range(n_bins):
        mask = digitized == bin_idx
        if not np.any(mask):
            continue
        bin_prob = y_prob[mask]
        bin_true = y_true[mask]
        bin_accuracy = bin_true.mean()
        bin_confidence = bin_prob.mean()
        ece += np.abs(bin_accuracy - bin_confidence) * mask.mean()
    return float(ece)


class CalibrationAnalyzer:
    """Perform calibration diagnostics and optional post-hoc correction."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        calibration_cfg = config.get("calibration", {})
        self.n_bins = calibration_cfg.get("n_bins", 10)

    def analyze_calibration(self, trained_models: Dict[str, Any], target_bundle: Any) -> Dict[str, pd.DataFrame]:
        results = []
        curve_frames = []
        calibrated_frames = []

        for model_name, model in trained_models.items():
            try:
                y_score_test = model.predict_proba(target_bundle.X_test)[:, 1]
                y_score_support = model.predict_proba(target_bundle.support_X)[:, 1] if target_bundle.support_X is not None else None
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("Calibration failed for %s: %s", model_name, exc)
                continue

            y_true_test = target_bundle.y_test
            metrics, threshold = compute_classification_metrics(y_true_test, y_score_test)
            hl_p = hosmer_lemeshow_test(y_true_test, y_score_test, self.n_bins)
            ece = expected_calibration_error(y_true_test, y_score_test, self.n_bins)
            brier = metrics.get("brier_score", np.nan)

            frac_pos, mean_pred = calibration_curve(y_true_test, y_score_test, n_bins=self.n_bins, strategy="quantile")
            curve_frames.append(
                pd.DataFrame({"mean_pred": mean_pred, "fraction_pos": frac_pos, "model": model_name})
            )

            results.append(
                {
                    "Model": model_name,
                    "BrierScore": brier,
                    "HosmerLemeshowP": hl_p,
                    "ECE": ece,
                    "OptimalThreshold": threshold,
                }
            )

            if y_score_support is None:
                continue

            calibration_methods = self.config.get("calibration", {}).get("methods", [])
            for method in calibration_methods:
                if method == "platt_scaling":
                    calibrator = LogisticRegression(max_iter=1000)
                    calibrator.fit(y_score_support.reshape(-1, 1), target_bundle.support_y)
                    calibrated_scores = calibrator.predict_proba(y_score_test.reshape(-1, 1))[:, 1]
                elif method == "isotonic_regression":
                    calibrator = IsotonicRegression(out_of_bounds="clip")
                    calibrator.fit(y_score_support, target_bundle.support_y)
                    calibrated_scores = calibrator.transform(y_score_test)
                else:
                    logger.warning("Unsupported calibration method %s", method)
                    continue

                calibrated_metrics, _ = compute_classification_metrics(y_true_test, calibrated_scores)
                calibrated_frames.append(
                    pd.DataFrame(
                        {
                            "Model": model_name,
                            "Method": method,
                            "Metric": list(calibrated_metrics.keys()),
                            "Value": list(calibrated_metrics.values()),
                        }
                    )
                )

        summary_df = pd.DataFrame(results)
        curves_df = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
        calibrated_df = pd.concat(calibrated_frames, ignore_index=True) if calibrated_frames else pd.DataFrame()
        return {
            "calibration_summary": summary_df,
            "calibration_curves": curves_df,
            "post_hoc_calibration": calibrated_df,
        }
