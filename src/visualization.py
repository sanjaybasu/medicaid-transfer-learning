"""Figure generation for Medicaid transfer learning analyses."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")


class VisualizationGenerator:
    """Create publication-ready plots for the manuscript update."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_figures(
        self,
        performance_results: Dict[str, pd.DataFrame],
        statistical_results: Dict[str, pd.DataFrame],
        calibration_results: Dict[str, pd.DataFrame],
        ablation_results: Dict[str, pd.DataFrame],
    ) -> None:
        try:
            self._plot_model_performance(performance_results.get("model_comparison", pd.DataFrame()))
            self._plot_sensitivity_specificity(performance_results.get("model_comparison", pd.DataFrame()))
            self._plot_calibration(calibration_results.get("calibration_curves", pd.DataFrame()))
            self._plot_feature_ablation(ablation_results.get("feature_ablation", pd.DataFrame()))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to generate figures: %s", exc)

    def _plot_model_performance(self, comparison_df: pd.DataFrame) -> None:
        if comparison_df.empty:
            logger.warning("Model comparison dataframe empty; skipping performance figure.")
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        ordered_df = comparison_df.sort_values(by="AUC", ascending=False)
        sns.barplot(
            data=ordered_df,
            x="Model",
            y="AUC",
            ax=ax,
            palette="viridis",
        )
        ax.set_ylabel("ROC AUC")
        ax.set_xlabel("")
        ax.set_ylim(0.4, max(0.65, ordered_df["AUC"].max() + 0.02))
        ax.set_title("Model discrimination across Medicaid domains")
        ax.tick_params(axis="x", rotation=20)
        for patch, value in zip(ax.patches, ordered_df["AUC"]):
            ax.annotate(f"{value:.3f}", (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                        ha="center", va="bottom", fontsize=9, xytext=(0, 3), textcoords="offset points")
        fig.tight_layout()
        fig_path = self.figures_dir / "figure_model_auc.png"
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
        logger.info("Saved %s", fig_path)

    def _plot_sensitivity_specificity(self, comparison_df: pd.DataFrame) -> None:
        if comparison_df.empty:
            return
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.scatterplot(
            data=comparison_df,
            x="Sensitivity",
            y="Specificity",
            hue="Model",
            style="Model",
            s=110,
            ax=ax
        )
        ax.set_xlabel("Sensitivity")
        ax.set_ylabel("Specificity")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("Sensitivity-specificity trade-off (optimal thresholds)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
        fig.subplots_adjust(right=0.72)
        fig_path = self.figures_dir / "figure_sensitivity_specificity.png"
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
        logger.info("Saved %s", fig_path)

    def _plot_calibration(self, calibration_df: pd.DataFrame) -> None:
        if calibration_df.empty:
            logger.warning("Calibration dataframe empty; skipping calibration plot.")
            return
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.lineplot(data=calibration_df, x="mean_pred", y="fraction_pos", hue="model", style="model", markers=True, dashes=False, ax=ax)
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
        ax.set_xlabel("Predicted risk")
        ax.set_ylabel("Observed outcome rate")
        ax.set_title("Calibration curve on target domain")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
        fig.subplots_adjust(right=0.72)
        fig_path = self.figures_dir / "figure_calibration.png"
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
        logger.info("Saved %s", fig_path)

    def _plot_feature_ablation(self, ablation_df: pd.DataFrame) -> None:
        if ablation_df.empty:
            logger.info("No ablation data to visualise.")
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=ablation_df, x="Category", y="AUC", palette="magma", ax=ax)
        ax.set_ylabel("ROC AUC")
        ax.set_xlabel("Feature subset")
        ax.set_ylim(0.4, max(0.7, ablation_df["AUC"].max() + 0.02))
        ax.tick_params(axis="x", rotation=30)
        ax.set_title("Feature group ablation on logistic baseline")
        fig.tight_layout()
        fig_path = self.figures_dir / "figure_feature_ablation.png"
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
        logger.info("Saved %s", fig_path)
