"""Evaluation utilities for Medicaid transfer learning experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelEvaluationResult:
    """Container for model-level evaluation artefacts."""

    name: str
    metrics: Dict[str, float]
    roc_curve: Dict[str, np.ndarray]
    pr_curve: Dict[str, np.ndarray]
    threshold: float


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def _optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    if thresholds.size == 0:
        return 0.5, float("nan"), float("nan")
    youdens = tpr - fpr
    idx = int(np.argmax(youdens))
    best_threshold = thresholds[idx]
    sensitivity = tpr[idx]
    specificity = 1 - fpr[idx]
    return float(best_threshold), float(sensitivity), float(specificity)


def compute_classification_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[Dict[str, float], float]:
    threshold, sensitivity, specificity = _optimal_threshold(y_true, y_score)
    y_pred = (y_score >= threshold).astype(int)

    precision_val = precision_score(y_true, y_pred, zero_division=0)
    recall_val = recall_score(y_true, y_pred, zero_division=0)
    f1_val = 2 * precision_val * recall_val / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0.0

    metrics = {
        "auc": _safe_auc(y_true, y_score),
        "average_precision": average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan"),
        "precision": precision_val,
        "recall": recall_val,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "youdens_j": sensitivity + specificity - 1 if not np.isnan(sensitivity) and not np.isnan(specificity) else float("nan"),
        "f1": f1_val,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float("nan"),
        "mcc": matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else float("nan"),
        "brier_score": brier_score_loss(y_true, y_score),
        "event_rate": float(np.mean(y_true)),
    }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() if len(np.unique(y_true)) > 1 else (0, 0, 0, 0)
    metrics.update(
        {
            "tp": float(tp),
            "fp": float(fp),
            "tn": float(tn),
            "fn": float(fn),
        }
    )

    return metrics, threshold


def compute_curve_data(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    if len(np.unique(y_true)) < 2:
        empty = {"x": np.array([]), "y": np.array([])}
        return empty, empty
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    if roc_thresholds.size and roc_thresholds.size < fpr.size:
        roc_thresholds = np.append(roc_thresholds, roc_thresholds[-1])
    if pr_thresholds.size and pr_thresholds.size < precision.size:
        pr_thresholds = np.append(pr_thresholds, pr_thresholds[-1])
    return (
        {"fpr": fpr, "tpr": tpr, "thresholds": roc_thresholds},
        {"precision": precision, "recall": recall, "thresholds": pr_thresholds},
    )


class EvaluationFramework:
    """Run evaluation across trained models and assemble comparison tables."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def evaluate_all_models(
        self,
        trained_models: Dict[str, Any],
        source_bundle: Any,
        target_bundle: Any,
    ) -> Dict[str, pd.DataFrame]:
        if not trained_models:
            raise ValueError("No models were trained; cannot evaluate.")

        model_results: List[ModelEvaluationResult] = []
        prediction_frames: List[pd.DataFrame] = []
        sample_indices = np.arange(target_bundle.X_test.shape[0])
        for model_name, model in trained_models.items():
            try:
                y_score = model.predict_proba(target_bundle.X_test)[:, 1]
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("Model %s failed during probability prediction: %s", model_name, exc)
                continue

            metrics, threshold = compute_classification_metrics(target_bundle.y_test, y_score)
            roc_data, pr_data = compute_curve_data(target_bundle.y_test, y_score)
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "y_true": target_bundle.y_test,
                        "y_score": y_score,
                        "model": model_name,
                        "sample_id": sample_indices,
                    }
                )
            )
            model_results.append(
                ModelEvaluationResult(
                    name=model_name,
                    metrics=metrics,
                    roc_curve=roc_data,
                    pr_curve=pr_data,
                    threshold=threshold,
                )
            )

        if not model_results:
            raise ValueError("No evaluation results were produced; check trained models.")

        def _format_metric_name(name: str) -> str:
            special = {"auc": "AUC", "mcc": "MCC"}
            if name in special:
                return special[name]
            parts = name.split("_")
            return "".join(part.capitalize() for part in parts)

        comparison_rows = []
        for res in model_results:
            metric_dict = {_format_metric_name(k): v for k, v in res.metrics.items()}
            row = {"Model": res.name, **metric_dict, "OptimalThreshold": res.threshold}
            comparison_rows.append(row)

        comparison_df = pd.DataFrame(comparison_rows).sort_values(by="AUC", ascending=False).reset_index(drop=True)

        roc_frames = []
        pr_frames = []
        for res in model_results:
            roc_df = pd.DataFrame({"fpr": res.roc_curve.get("fpr", []), "tpr": res.roc_curve.get("tpr", []), "threshold": res.roc_curve.get("thresholds", [])})
            roc_df["model"] = res.name
            pr_df = pd.DataFrame({"precision": res.pr_curve.get("precision", []), "recall": res.pr_curve.get("recall", []), "threshold": res.pr_curve.get("thresholds", [])})
            pr_df["model"] = res.name
            roc_frames.append(roc_df)
            pr_frames.append(pr_df)

        roc_summary = pd.concat(roc_frames, ignore_index=True) if roc_frames else pd.DataFrame()
        pr_summary = pd.concat(pr_frames, ignore_index=True) if pr_frames else pd.DataFrame()

        predictions_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()

        return {
            "model_comparison": comparison_df,
            "roc_curves": roc_summary,
            "pr_curves": pr_summary,
            "predictions": predictions_df,
        }
