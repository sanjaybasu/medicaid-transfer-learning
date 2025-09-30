"""Bootstrap-based statistical evaluation for model comparison."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from evaluation_metrics import compute_classification_metrics, _safe_auc

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Run bootstrap hypothesis tests over evaluation outputs."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        bootstrap_cfg = config.get("bootstrap", {})
        self.n_iterations = bootstrap_cfg.get("n_iterations", 1000)
        self.confidence_level = bootstrap_cfg.get("confidence_level", 0.95)
        self.random_state = config.get("random_seeds", {}).get("bootstrap", 1234)
        self.rng = np.random.default_rng(self.random_state)

    def perform_statistical_tests(self, evaluation_outputs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        predictions = evaluation_outputs.get("predictions")
        if predictions is None or predictions.empty:
            logger.warning("Prediction-level data absent; skipping statistical analysis.")
            return {"significance_tests": pd.DataFrame()}

        model_names = predictions["model"].unique()
        baseline_models = self.config.get("evaluation", {}).get("baseline_models", ["source_only", "target_only"])
        rows: List[Dict[str, Any]] = []

        for baseline in baseline_models:
            if baseline not in model_names:
                logger.warning("Baseline model %s not present; skipping.", baseline)
                continue
            for comparator in model_names:
                if comparator == baseline:
                    continue
                rows.append(self._compare_models(predictions, comparator, baseline))

        results_df = pd.DataFrame(rows)
        return {"significance_tests": results_df}

    def _compare_models(self, predictions: pd.DataFrame, model_a: str, model_b: str) -> Dict[str, Any]:
        merged = predictions[predictions["model"] == model_a].merge(
            predictions[predictions["model"] == model_b],
            on="sample_id",
            suffixes=("_a", "_b"),
        )
        if merged.empty:
            logger.warning("Unable to align predictions for %s vs %s", model_a, model_b)
            return {
                "Comparison": f"{model_a} vs {model_b}",
                "metric": "auc",
                "difference": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "p_value": np.nan,
            }

        y_true = merged["y_true_a"].to_numpy()
        scores_a = merged["y_score_a"].to_numpy()
        scores_b = merged["y_score_b"].to_numpy()

        auc_results = self._bootstrap_metric(y_true, scores_a, scores_b, metric="auc")
        youden_results = self._bootstrap_metric(y_true, scores_a, scores_b, metric="youdens_j")

        return {
            "Comparison": f"{model_a} vs {model_b}",
            "metric": "auc",
            "difference": auc_results["difference"],
            "ci_lower": auc_results["ci_lower"],
            "ci_upper": auc_results["ci_upper"],
            "p_value": auc_results["p_value"],
            "youdens_diff": youden_results["difference"],
            "youdens_ci_lower": youden_results["ci_lower"],
            "youdens_ci_upper": youden_results["ci_upper"],
        }

    def _bootstrap_metric(self, y_true: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray, metric: str) -> Dict[str, float]:
        n_samples = y_true.shape[0]
        if n_samples == 0:
            return {"difference": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "p_value": np.nan}

        diffs = []
        for _ in range(self.n_iterations):
            sample_idx = self.rng.integers(0, n_samples, size=n_samples)
            y_sample = y_true[sample_idx]
            scores_a_sample = scores_a[sample_idx]
            scores_b_sample = scores_b[sample_idx]

            if metric == "auc":
                metric_a = _safe_auc(y_sample, scores_a_sample)
                metric_b = _safe_auc(y_sample, scores_b_sample)
            elif metric == "youdens_j":
                metrics_a, _ = compute_classification_metrics(y_sample, scores_a_sample)
                metrics_b, _ = compute_classification_metrics(y_sample, scores_b_sample)
                metric_a = metrics_a.get("youdens_j", np.nan)
                metric_b = metrics_b.get("youdens_j", np.nan)
            else:
                raise ValueError(f"Unsupported metric {metric}")

            diffs.append(metric_a - metric_b)

        diffs_arr = np.array(diffs)
        alpha = 1 - self.confidence_level
        ci_lower, ci_upper = np.quantile(diffs_arr, [alpha / 2, 1 - alpha / 2])
        p_value = 2 * min(np.mean(diffs_arr <= 0), np.mean(diffs_arr >= 0))
        return {"difference": float(np.mean(diffs_arr)), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "p_value": float(p_value)}
