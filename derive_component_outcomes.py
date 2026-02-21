#!/usr/bin/env python3
"""
Component-outcome derivation script for Medicaid transfer learning analysis.

Builds ed_utilization_12m and hosp_utilization_12m columns for the Virginia and
Washington analysis datasets using locally available raw data files:
  - outcomes_monthly.csv  (emergency_department_ct, acute_inpatient_ct per member per month)
  - eligibility.csv       (demographics, enrollment dates, state)
  - member_attributes.csv (gender, birth_date, race/ethnicity)

Cohort definition mirrors aws_metalearning_revision.ipynb:
  - Index date: 2024-06-01
  - Follow-up window: 2024-06-01 to 2025-05-31 (12 months)
  - Baseline window:  2023-06-01 to 2024-05-31 (12 months)
  - Age 18–64 at index date
  - VA payer (ABHVA or SHPVA); WA payer (UHCWA, PROVWA, CHPW)

NOTE: Comorbidity flags (diabetes, hypertension, etc.), charlson_comorbidity_index,
medication_count, urban_rural, county_type, and health_service_area are not available
in local raw files and are set to 0.  These will be filled by the pipeline with median
imputation but will not affect outcome columns.  The primary analysis results (AUC,
calibration) in the paper were obtained from the SageMaker-exported CSVs that include
these features; this script is intended only to populate component outcome columns and
run the component_outcome_analysis.py module.

Usage:
    python derive_component_outcomes.py [--data-dir DATA_DIR] [--out-dir OUT_DIR]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Window constants ──────────────────────────────────────────────────────────
INDEX_DATE = pd.Timestamp("2024-06-01")
FOLLOWUP_END = pd.Timestamp("2025-05-31")
BASELINE_START = pd.Timestamp("2023-06-01")
BASELINE_END = INDEX_DATE - pd.Timedelta(days=1)

# ── Payer → state mapping ─────────────────────────────────────────────────────
VA_PAYERS = {"ABHVA", "SHPVA"}
WA_PAYERS = {"UHCWA", "PROVWA", "CHPW"}


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Fall"


def load_raw_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log.info("Loading raw data files from %s", data_dir)

    om = pd.read_csv(data_dir / "outcomes_monthly.csv", low_memory=False)
    om["month_year_dt"] = pd.to_datetime(om["month_year"] + "-01")
    log.info("  outcomes_monthly: %d rows", len(om))

    elig = pd.read_csv(data_dir / "eligibility.csv", low_memory=False)
    elig["birth_date"] = pd.to_datetime(elig["birth_date"], errors="coerce")
    elig["enrollment_start_date"] = pd.to_datetime(elig["enrollment_start_date"], errors="coerce")
    elig["enrollment_end_date"] = pd.to_datetime(elig["enrollment_end_date"], errors="coerce")
    log.info("  eligibility: %d rows", len(elig))

    ma = pd.read_csv(data_dir / "member_attributes.csv", low_memory=False)
    log.info("  member_attributes: %d rows", len(ma))

    return om, elig, ma


def build_cohort(elig: pd.DataFrame, state: str) -> pd.DataFrame:
    """Return one row per person in the care-management cohort for *state*."""
    payers = VA_PAYERS if state == "VA" else WA_PAYERS

    df = elig[elig["payer"].isin(payers)].copy()

    # Age at index date
    df["age"] = (INDEX_DATE - df["birth_date"]).dt.days / 365.25

    # Apply cohort criteria: age 18–64, not dual-eligible
    df = df[df["age"].between(18, 64, inclusive="both")]
    if "dual_status_code" in df.columns:
        df = df[df["dual_status_code"].isna() | (df["dual_status_code"] == 0)]

    # Deduplicate: keep most recent enrollment record per person
    df = (
        df.sort_values("enrollment_end_date", ascending=False)
        .drop_duplicates(subset="person_id", keep="first")
        .reset_index(drop=True)
    )

    log.info("  %s cohort: %d members", state, len(df))
    return df


def compute_baseline_utilization(om: pd.DataFrame, person_ids: set) -> pd.DataFrame:
    """Sum utilization in baseline window per person_id."""
    mask = (
        om["person_id"].isin(person_ids)
        & (om["month_year_dt"] >= BASELINE_START)
        & (om["month_year_dt"] <= BASELINE_END)
    )
    base = om[mask].groupby("person_id").agg(
        prior_ed_visits_12m=("emergency_department_ct", "sum"),
        prior_hospitalizations_12m=("acute_inpatient_ct", "sum"),
        prior_outpatient_visits_12m=("outpatient_hospital_or_clinic_ct", "sum"),
    ).reset_index()
    return base


def compute_followup_outcomes(om: pd.DataFrame, person_ids: set) -> pd.DataFrame:
    """Compute binary component outcomes in follow-up window per person_id."""
    mask = (
        om["person_id"].isin(person_ids)
        & (om["month_year_dt"] >= INDEX_DATE)
        & (om["month_year_dt"] <= FOLLOWUP_END)
    )
    fu = om[mask].groupby("person_id").agg(
        ed_events=("emergency_department_ct", "sum"),
        hosp_events=("acute_inpatient_ct", "sum"),
    ).reset_index()
    fu["ed_utilization_12m"] = (fu["ed_events"] > 0).astype(int)
    fu["hosp_utilization_12m"] = (fu["hosp_events"] > 0).astype(int)
    fu["acute_care_utilization_12m"] = ((fu["ed_events"] + fu["hosp_events"]) > 0).astype(int)
    return fu[["person_id", "ed_utilization_12m", "hosp_utilization_12m", "acute_care_utilization_12m"]]


def build_state_dataset(
    state: str,
    cohort: pd.DataFrame,
    om: pd.DataFrame,
    ma: pd.DataFrame,
) -> pd.DataFrame:
    person_ids = set(cohort["person_id"])

    # Baseline utilization
    log.info("  Computing baseline utilization for %s...", state)
    baseline = compute_baseline_utilization(om, person_ids)

    # Follow-up outcomes
    log.info("  Computing follow-up outcomes for %s...", state)
    outcomes = compute_followup_outcomes(om, person_ids)

    # Enrollment features from eligibility
    cohort_features = cohort[["person_id", "age", "gender", "race", "county",
                               "total_member_months_covered", "enrollment_start_date",
                               "enrollment_end_date", "payer"]].copy()

    # Merge member_attributes for race/ethnicity and gender (may be more complete)
    ma_slim = ma[["waymark_patient_number", "gender", "race", "ethnicity"]].rename(
        columns={"waymark_patient_number": "person_id"}
    )
    # Use eligibility gender/race first; fill from member_attributes if missing
    cohort_features = cohort_features.merge(ma_slim, on="person_id", how="left",
                                             suffixes=("_elig", "_ma"))
    cohort_features["gender"] = cohort_features["gender_elig"].fillna(cohort_features["gender_ma"])
    cohort_features["race_ethnicity"] = cohort_features["race_elig"].fillna(cohort_features["race_ma"])
    cohort_features = cohort_features.drop(columns=["gender_elig", "gender_ma",
                                                      "race_elig", "race_ma"], errors="ignore")

    # Enrollment duration (months) and index month
    cohort_features["enrollment_duration"] = (
        cohort_features["total_member_months_covered"]
        .fillna(
            ((cohort_features["enrollment_end_date"].fillna(INDEX_DATE) -
              cohort_features["enrollment_start_date"].fillna(INDEX_DATE)).dt.days / 30.4)
            .clip(lower=1)
        )
        .clip(lower=1)
    ).astype(int)

    cohort_features["enrollment_month"] = (
        cohort_features["enrollment_start_date"]
        .dt.month
        .fillna(INDEX_DATE.month)
        .astype(int)
    )
    cohort_features["seasonal_indicators"] = cohort_features["enrollment_month"].apply(_season)
    cohort_features["managed_care_enrollment"] = 1  # All are in managed care (care management enrolled)
    cohort_features["index_date"] = INDEX_DATE
    cohort_features["state"] = state
    cohort_features["time_trends"] = 0  # Not derivable locally

    # Eligibility category proxy from payer
    cohort_features["eligibility_category"] = cohort_features["payer"]

    # Features not available locally → 0
    for col in ["urban_rural", "diabetes_mellitus", "hypertension", "heart_disease", "copd",
                "mental_health_disorders", "substance_abuse_disorders",
                "charlson_comorbidity_index", "medication_count",
                "county_type", "health_service_area"]:
        cohort_features[col] = 0

    # Merge baseline utilization
    dataset = cohort_features.merge(baseline, on="person_id", how="left").fillna(
        {"prior_ed_visits_12m": 0, "prior_hospitalizations_12m": 0, "prior_outpatient_visits_12m": 0}
    )
    dataset[["prior_ed_visits_12m", "prior_hospitalizations_12m",
             "prior_outpatient_visits_12m"]] = dataset[[
        "prior_ed_visits_12m", "prior_hospitalizations_12m", "prior_outpatient_visits_12m"
    ]].astype(int)

    # Merge outcomes
    dataset = dataset.merge(outcomes, on="person_id", how="left").fillna(
        {"ed_utilization_12m": 0, "hosp_utilization_12m": 0, "acute_care_utilization_12m": 0}
    )
    dataset[["ed_utilization_12m", "hosp_utilization_12m",
             "acute_care_utilization_12m"]] = dataset[[
        "ed_utilization_12m", "hosp_utilization_12m", "acute_care_utilization_12m"
    ]].astype(int)

    # Rename person_id to patient_id for pipeline compatibility
    dataset = dataset.rename(columns={"person_id": "patient_id"})

    # Log outcome prevalences
    n = len(dataset)
    log.info(
        "%s (N=%d) — composite: %.3f, ED-only: %.3f, hosp-only: %.3f",
        state, n,
        dataset["acute_care_utilization_12m"].mean(),
        dataset["ed_utilization_12m"].mean(),
        dataset["hosp_utilization_12m"].mean(),
    )
    return dataset


# ── Column order matching pipeline expectations ───────────────────────────────
COLUMN_ORDER = [
    "patient_id", "state", "index_date", "age", "gender", "race_ethnicity", "urban_rural",
    "diabetes_mellitus", "hypertension", "heart_disease", "copd",
    "mental_health_disorders", "substance_abuse_disorders", "charlson_comorbidity_index",
    "prior_ed_visits_12m", "prior_hospitalizations_12m", "prior_outpatient_visits_12m",
    "medication_count", "eligibility_category", "enrollment_duration",
    "managed_care_enrollment", "county_type", "health_service_area",
    "enrollment_month", "seasonal_indicators", "time_trends",
    "acute_care_utilization_12m", "ed_utilization_12m", "hosp_utilization_12m",
]


def main(data_dir: Path, out_dir: Path) -> None:
    om, elig, ma = load_raw_data(data_dir)

    for state in ("VA", "WA"):
        log.info("=== Building %s dataset ===", state)
        cohort = build_cohort(elig, state)
        dataset = build_state_dataset(state, cohort, om, ma)

        # Select and order columns
        present = [c for c in COLUMN_ORDER if c in dataset.columns]
        dataset = dataset[present]

        filename = "virginia_medicaid.csv" if state == "VA" else "washington_medicaid.csv"
        out_path = out_dir / filename
        dataset.to_csv(out_path, index=False)
        log.info("Wrote %d rows to %s", len(dataset), out_path)

    log.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="/Users/sanjaybasu/waymark-local/data/real_inputs",
        help="Directory containing raw input CSV files",
    )
    parser.add_argument(
        "--out-dir",
        default="data",
        help="Output directory for analysis CSVs (default: data/)",
    )
    args = parser.parse_args()
    main(Path(args.data_dir), Path(args.out_dir))
