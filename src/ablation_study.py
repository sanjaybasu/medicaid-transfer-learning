"""Feature and component ablation utilities for transfer learning experiments."""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from evaluation_metrics import compute_classification_metrics

logger = logging.getLogger(__name__)


class AblationStudy:
    """Run feature-group and model-component ablations."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def perform_ablation_analysis(
        self,
        source_bundle: Any,
        target_bundle: Any,
        trained_models: Optional[Dict[str, Any]] = None,
        performance_results: Optional[pd.DataFrame] = None,
    ) -> Dict[str, pd.DataFrame]:
        feature_results = self._feature_category_ablation(source_bundle, target_bundle)
        component_results = self._component_ablation(
            source_bundle, target_bundle, trained_models, performance_results
        )
        return {
            "feature_ablation": feature_results,
            "component_ablation": component_results,
        }

    def _feature_category_ablation(self, source_bundle: Any, target_bundle: Any) -> pd.DataFrame:
        category_map = source_bundle.feature_category_map
        feature_names = source_bundle.feature_names

        baseline_metrics = self._fit_logistic_and_score(
            source_bundle.X_train,
            source_bundle.y_train,
            target_bundle.X_test,
            target_bundle.y_test,
        )
        records: List[Dict[str, Any]] = [
            {
                "Category": "All Features",
                "AUC": baseline_metrics["auc"],
                "Youdens_J": baseline_metrics["youdens_j"],
                "F1": baseline_metrics["f1"],
            }
        ]

        unique_categories = sorted({category_map.get(name, "other") for name in feature_names})
        for category in unique_categories:
            mask = np.array([category_map.get(name, "other") != category for name in feature_names])
            if not mask.any():
                continue
            metrics = self._fit_logistic_and_score(
                source_bundle.X_train[:, mask],
                source_bundle.y_train,
                target_bundle.X_test[:, mask],
                target_bundle.y_test,
            )
            records.append(
                {
                    "Category": f"Drop {category}",
                    "AUC": metrics["auc"],
                    "Youdens_J": metrics["youdens_j"],
                    "F1": metrics["f1"],
                }
            )

        return pd.DataFrame(records)

    def _fit_logistic_and_score(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = LogisticRegression(max_iter=1000, solver="lbfgs")
        model.fit(X_train_scaled, y_train)
        y_score = model.predict_proba(X_test_scaled)[:, 1]
        metrics, _ = compute_classification_metrics(y_test, y_score)
        return metrics

    def _component_ablation(
        self,
        source_bundle: Any,
        target_bundle: Any,
        trained_models: Optional[Dict[str, Any]],
        performance_results: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        component_names = self.config.get("ablation", {}).get("model_components", [])
        if not component_names:
            return pd.DataFrame()

        try:
            from transfer_learning_models import EnhancedMAML
        except ImportError:  # pragma: no cover - defensive safeguard
            logger.error("EnhancedMAML implementation missing; skipping component ablation")
            return pd.DataFrame()

        if performance_results is not None:
            mask = performance_results["Model"] == "enhanced_maml"
            if mask.any():
                baseline_row = performance_results.loc[mask].iloc[0]
                base_metrics = {
                    "auc": baseline_row["AUC"],
                    "youdens_j": baseline_row["YoudensJ"],
                    "f1": baseline_row["F1"],
                }
            else:
                base_metrics = None
        else:
            base_metrics = None

        if base_metrics is None and trained_models and "enhanced_maml" in trained_models:
            base_model = trained_models["enhanced_maml"]
            base_probs = base_model.predict_proba(target_bundle.X_test)[:, 1]
            base_metrics, _ = compute_classification_metrics(target_bundle.y_test, base_probs)
        if base_metrics is None:
            base_model = EnhancedMAML(self.config)
            base_model.fit(source_bundle, target_bundle)
            base_metrics, _ = compute_classification_metrics(
                target_bundle.y_test, base_model.predict_proba(target_bundle.X_test)[:, 1]
            )

        records: List[Dict[str, Any]] = [
            {
                "Configuration": "All Components",
                "AUC": base_metrics["auc"],
                "Youdens_J": base_metrics["youdens_j"],
                "F1": base_metrics["f1"],
            }
        ]

        for component in component_names:
            override_config = copy.deepcopy(self.config)
            override_config.setdefault("models", {}).setdefault("enhanced_maml", {})[component] = False
            model = EnhancedMAML(override_config)
            model.fit(source_bundle, target_bundle)
            metrics, _ = compute_classification_metrics(
                target_bundle.y_test, model.predict_proba(target_bundle.X_test)[:, 1]
            )
            records.append(
                {
                    "Configuration": f"Without {component}",
                    "AUC": metrics["auc"],
                    "Youdens_J": metrics["youdens_j"],
                    "F1": metrics["f1"],
                }
            )

        return pd.DataFrame(records)
