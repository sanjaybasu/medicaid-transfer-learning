"""Fairness evaluation for transfer learning models.

Computes equalized odds across demographic subgroups (race/ethnicity, gender, age)
for each model's predictions on the Virginia hold-out test set.

Equalized odds requires that true positive rate (TPR) and false positive rate (FPR)
are equal across groups. We report the maximum absolute difference in TPR and FPR
across subgroups as the equalized odds difference (EOD).

Reference:
    Hardt M, Price E, Srebro N. Equality of opportunity in supervised learning.
    Proc NeurIPS. 2016;29:3315-3323.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

logger = logging.getLogger(__name__)


def _optimal_threshold_for_youden(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Return the threshold that maximises Youden's J."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = int(np.argmax(tpr - fpr))
    return float(thresholds[idx])


def _subgroup_tpr_fpr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float, int]:
    """Compute TPR and FPR for a boolean mask subgroup."""
    yt = y_true[mask]
    yp = y_pred[mask]
    n = int(mask.sum())
    if n == 0 or len(np.unique(yt)) < 2:
        return float("nan"), float("nan"), n
    tp = int(((yp == 1) & (yt == 1)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum())
    tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    return tpr, fpr, n


def compute_equalized_odds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    demographics: pd.DataFrame,
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compute equalized odds differences across demographic subgroups.

    Parameters
    ----------
    y_true : array of shape (n,)
        Binary true labels.
    y_score : array of shape (n,)
        Predicted probabilities.
    demographics : DataFrame of shape (n, ...)
        Must contain columns: race_ethnicity, gender, age.
        Index must align with y_true / y_score.
    threshold : float, optional
        Decision threshold. If None, uses the Youden-optimal threshold
        derived from y_true and y_score.

    Returns
    -------
    DataFrame with columns:
        dimension, subgroup, n, tpr, fpr, tpr_diff, fpr_diff, eod
    """
    if threshold is None:
        threshold = _optimal_threshold_for_youden(y_true, y_score)

    y_pred = (y_score >= threshold).astype(int)
    rows: List[Dict[str, Any]] = []

    # ── Race/ethnicity ────────────────────────────────────────────────────────
    # EOD = max pairwise |ΔTPR| or |ΔFPR| across all observed groups with
    # sufficient representation (n >= 30 and both positive/negative labels).
    if "race_ethnicity" in demographics.columns:
        observed_groups = sorted(demographics["race_ethnicity"].dropna().unique().tolist())
        race_tprs: Dict[str, float] = {}
        race_fprs: Dict[str, float] = {}
        for grp in observed_groups:
            mask = (demographics["race_ethnicity"] == grp).to_numpy()
            tpr, fpr, n = _subgroup_tpr_fpr(y_true, y_pred, mask)
            race_tprs[grp] = tpr
            race_fprs[grp] = fpr
            rows.append(
                {
                    "dimension": "race_ethnicity",
                    "subgroup": grp,
                    "n": n,
                    "tpr": round(tpr, 4) if not np.isnan(tpr) else np.nan,
                    "fpr": round(fpr, 4) if not np.isnan(fpr) else np.nan,
                    "tpr_diff_vs_ref": np.nan,
                    "fpr_diff_vs_ref": np.nan,
                    "eod": np.nan,
                }
            )
        # Compute max pairwise |ΔTPR| and |ΔFPR| across all groups as overall EOD.
        valid_tprs = [v for v in race_tprs.values() if not np.isnan(v)]
        valid_fprs = [v for v in race_fprs.values() if not np.isnan(v)]
        max_tpr_diff = max(valid_tprs) - min(valid_tprs) if len(valid_tprs) >= 2 else float("nan")
        max_fpr_diff = max(valid_fprs) - min(valid_fprs) if len(valid_fprs) >= 2 else float("nan")
        overall_eod = max(max_tpr_diff, max_fpr_diff) if not np.isnan(max_tpr_diff) else float("nan")
        for row in rows:
            if row["dimension"] == "race_ethnicity":
                row["eod"] = round(overall_eod, 4) if not np.isnan(overall_eod) else np.nan

    # ── Gender ────────────────────────────────────────────────────────────────
    if "gender" in demographics.columns:
        for grp in ["Female", "Male"]:
            mask = (demographics["gender"] == grp).to_numpy()
            tpr, fpr, n = _subgroup_tpr_fpr(y_true, y_pred, mask)
            rows.append(
                {
                    "dimension": "gender",
                    "subgroup": grp,
                    "n": n,
                    "tpr": round(tpr, 4) if not np.isnan(tpr) else np.nan,
                    "fpr": round(fpr, 4) if not np.isnan(fpr) else np.nan,
                    "tpr_diff_vs_white": np.nan,
                    "fpr_diff_vs_white": np.nan,
                    "eod": np.nan,
                }
            )
        # Compute EOD for gender: Female vs Male
        gender_rows = [r for r in rows if r["dimension"] == "gender"]
        tprs = {r["subgroup"]: r["tpr"] for r in gender_rows}
        fprs = {r["subgroup"]: r["fpr"] for r in gender_rows}
        tpr_diff = abs(tprs.get("Female", float("nan")) - tprs.get("Male", float("nan")))
        fpr_diff = abs(fprs.get("Female", float("nan")) - fprs.get("Male", float("nan")))
        eod = max(tpr_diff, fpr_diff)
        for row in gender_rows:
            row["eod"] = round(eod, 4) if not np.isnan(eod) else np.nan

    # ── Age group ─────────────────────────────────────────────────────────────
    if "age" in demographics.columns:
        age = demographics["age"].to_numpy()
        age_bins = [("lt30", age < 30), ("30to50", (age >= 30) & (age <= 50)), ("gt50", age > 50)]
        age_tprs: Dict[str, float] = {}
        age_fprs: Dict[str, float] = {}
        for label, mask in age_bins:
            tpr, fpr, n = _subgroup_tpr_fpr(y_true, y_pred, mask)
            age_tprs[label] = tpr
            age_fprs[label] = fpr
            rows.append(
                {
                    "dimension": "age_group",
                    "subgroup": label,
                    "n": n,
                    "tpr": round(tpr, 4) if not np.isnan(tpr) else np.nan,
                    "fpr": round(fpr, 4) if not np.isnan(fpr) else np.nan,
                    "tpr_diff_vs_white": np.nan,
                    "fpr_diff_vs_white": np.nan,
                    "eod": np.nan,
                }
            )
        # EOD = max absolute difference across any pair of age groups
        valid_tprs = [v for v in age_tprs.values() if not np.isnan(v)]
        valid_fprs = [v for v in age_fprs.values() if not np.isnan(v)]
        max_tpr_diff = max(valid_tprs) - min(valid_tprs) if len(valid_tprs) >= 2 else float("nan")
        max_fpr_diff = max(valid_fprs) - min(valid_fprs) if len(valid_fprs) >= 2 else float("nan")
        eod = max(max_tpr_diff, max_fpr_diff) if not np.isnan(max_tpr_diff) else float("nan")
        for row in rows:
            if row["dimension"] == "age_group":
                row["eod"] = round(eod, 4) if not np.isnan(eod) else np.nan

    return pd.DataFrame(rows)


def compute_fairness_summary(
    predictions: pd.DataFrame,
    demographics: pd.DataFrame,
    thresholds: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Compute equalized odds for every model in the predictions DataFrame.

    Parameters
    ----------
    predictions : DataFrame with columns y_true, y_score, model, sample_id
    demographics : DataFrame indexed by sample_id with race_ethnicity, gender, age
    thresholds : dict of model_name -> threshold (Youden-optimal per model)

    Returns
    -------
    Wide-format DataFrame with one row per model and EOD columns.
    """
    model_names = predictions["model"].unique()
    summary_rows: List[Dict[str, Any]] = []

    for model_name in model_names:
        model_preds = predictions[predictions["model"] == model_name].copy()
        # Align demographics to test set sample_ids
        aligned_demo = demographics.loc[model_preds["sample_id"].values] if demographics.index.name == "sample_id" else demographics.iloc[model_preds["sample_id"].values]

        threshold = None
        if thresholds is not None:
            threshold = thresholds.get(model_name)

        detail_df = compute_equalized_odds(
            y_true=model_preds["y_true"].to_numpy(),
            y_score=model_preds["y_score"].to_numpy(),
            demographics=aligned_demo.reset_index(drop=True),
            threshold=threshold,
        )

        row: Dict[str, Any] = {"Model": model_name}

        # Race/ethnicity: overall EOD (max pairwise TPR or FPR difference)
        race_rows = detail_df[detail_df["dimension"] == "race_ethnicity"]
        race_eods = race_rows["eod"].dropna()
        row["EOD_race_ethnicity"] = round(float(race_eods.iloc[0]), 4) if len(race_eods) > 0 else np.nan

        # Group with lowest TPR (worst-served group)
        race_tpr_rows = race_rows[race_rows["tpr"].notna()]
        if not race_tpr_rows.empty:
            worst_idx = race_tpr_rows["tpr"].idxmin()
            row["worst_race_group"] = race_tpr_rows.loc[worst_idx, "subgroup"]
            row["min_TPR_race"] = round(float(race_tpr_rows.loc[worst_idx, "tpr"]), 4)
        else:
            row["worst_race_group"] = np.nan
            row["min_TPR_race"] = np.nan

        # Gender EOD
        gender_eods = detail_df[detail_df["dimension"] == "gender"]["eod"].dropna()
        row["EOD_gender"] = round(float(gender_eods.iloc[0]), 4) if len(gender_eods) > 0 else np.nan

        # Age EOD
        age_eods = detail_df[detail_df["dimension"] == "age_group"]["eod"].dropna()
        row["EOD_age_group"] = round(float(age_eods.iloc[0]), 4) if len(age_eods) > 0 else np.nan

        # Per-subgroup TPRs for observed groups with n >= 30 (for display in paper)
        race_rows_all = detail_df[detail_df["dimension"] == "race_ethnicity"]
        for _, grp_row in race_rows_all.iterrows():
            grp = grp_row["subgroup"]
            if grp_row["n"] >= 30:
                row[f"TPR_{grp}"] = round(float(grp_row["tpr"]), 4) if not np.isnan(grp_row["tpr"]) else np.nan
                row[f"FPR_{grp}"] = round(float(grp_row["fpr"]), 4) if not np.isnan(grp_row["fpr"]) else np.nan
                row[f"n_{grp}"] = int(grp_row["n"])

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


class FairnessAnalyzer:
    """Orchestrate fairness evaluation across all models."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def analyze_fairness(
        self,
        predictions: pd.DataFrame,
        target_demographics: pd.DataFrame,
        model_thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Run fairness evaluation for all models.

        Parameters
        ----------
        predictions : DataFrame with columns y_true, y_score, model, sample_id
        target_demographics : DataFrame with race_ethnicity, gender, age columns
            (one row per test patient, index = 0..n_test-1)
        model_thresholds : optional dict of model_name -> decision threshold

        Returns
        -------
        dict with keys 'fairness_summary' and 'fairness_detail'
        """
        if predictions is None or predictions.empty:
            logger.warning("No predictions available for fairness analysis.")
            return {"fairness_summary": pd.DataFrame(), "fairness_detail": pd.DataFrame()}

        required_cols = {"race_ethnicity", "gender", "age"}
        missing = required_cols - set(target_demographics.columns)
        if missing:
            logger.warning("Demographics missing columns %s; skipping fairness analysis.", missing)
            return {"fairness_summary": pd.DataFrame(), "fairness_detail": pd.DataFrame()}

        logger.info("Computing equalized odds across %d models.", predictions["model"].nunique())

        summary = compute_fairness_summary(predictions, target_demographics, model_thresholds)
        summary = summary.sort_values("EOD_race_ethnicity", ascending=True, na_position="last").reset_index(drop=True)

        # Detailed per-subgroup breakdown for supplementary
        detail_rows = []
        for model_name in predictions["model"].unique():
            model_preds = predictions[predictions["model"] == model_name].copy()
            aligned = target_demographics.iloc[model_preds["sample_id"].values].reset_index(drop=True)
            threshold = model_thresholds.get(model_name) if model_thresholds else None
            detail = compute_equalized_odds(
                model_preds["y_true"].to_numpy(),
                model_preds["y_score"].to_numpy(),
                aligned,
                threshold=threshold,
            )
            detail.insert(0, "Model", model_name)
            detail_rows.append(detail)

        detail_df = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()

        return {"fairness_summary": summary, "fairness_detail": detail_df}
