"""Data loading and preprocessing utilities for Medicaid transfer learning study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@dataclass
class DomainDataBundle:
    """Container for processed domain-specific datasets."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_full: np.ndarray
    y_full: np.ndarray
    feature_names: List[str]
    feature_category_map: Dict[str, str]
    metadata: Dict[str, Any]
    support_X: Optional[np.ndarray] = None
    support_y: Optional[np.ndarray] = None
    query_X: Optional[np.ndarray] = None
    query_y: Optional[np.ndarray] = None
    # Demographic columns (race_ethnicity, gender, age) for test-set patients —
    # used by fairness analysis. Index-aligned to X_test rows.
    test_demographics: Optional[pd.DataFrame] = None
    # Raw (pre-encoding) domain DataFrame, row-aligned; used by component-outcome
    # analysis to access ed_utilization_12m and hosp_utilization_12m columns.
    raw_df: Optional[pd.DataFrame] = None
    # Integer indices into raw_df that correspond to X_test rows.
    test_indices: Optional[np.ndarray] = None


class SyntheticDataGenerator:
    """Synthetic Medicaid-like dataset generator for local experimentation."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def generate_state_population(self, n_samples: int, state: str) -> pd.DataFrame:
        """Generate synthetic population for a single state with mild domain shift."""

        age = self.rng.integers(18, 65, size=n_samples)
        gender = self.rng.choice(["Female", "Male"], size=n_samples, p=[0.58, 0.42])
        race_ethnicity = self.rng.choice(
            ["White", "Black", "Hispanic", "Other"],
            size=n_samples,
            p=[0.45, 0.35, 0.12, 0.08] if state == "WA" else [0.38, 0.42, 0.10, 0.10],
        )
        urban_rural = self.rng.choice(["Urban", "Rural"], size=n_samples, p=[0.7, 0.3] if state == "WA" else [0.55, 0.45])

        diabetes = self.rng.binomial(1, 0.28 if state == "WA" else 0.32, size=n_samples)
        hypertension = self.rng.binomial(1, 0.42 if state == "WA" else 0.48, size=n_samples)
        heart_disease = self.rng.binomial(1, 0.18 if state == "WA" else 0.22, size=n_samples)
        copd = self.rng.binomial(1, 0.16 if state == "WA" else 0.20, size=n_samples)
        mental_health = self.rng.binomial(1, 0.35 if state == "WA" else 0.31, size=n_samples)
        substance_use = self.rng.binomial(1, 0.19 if state == "WA" else 0.24, size=n_samples)

        charlson_index = (
            diabetes + hypertension + heart_disease + copd + self.rng.poisson(1.5, size=n_samples)
        )
        charlson_index = np.clip(charlson_index, 0, None)

        prior_ed = self.rng.poisson(2.1 if state == "WA" else 2.5, size=n_samples)
        prior_hosp = self.rng.poisson(0.6 if state == "WA" else 0.8, size=n_samples)
        prior_outpatient = self.rng.poisson(14 if state == "WA" else 12, size=n_samples)
        medication_count = self.rng.poisson(8 if state == "WA" else 9, size=n_samples)

        eligibility = self.rng.choice(
            ["Disabled", "TANF", "Expansion", "Other"],
            size=n_samples,
            p=[0.32, 0.30, 0.28, 0.10] if state == "WA" else [0.36, 0.26, 0.28, 0.10],
        )
        enrollment_duration = self.rng.integers(12, 61, size=n_samples)
        managed_care = self.rng.binomial(1, 0.78 if state == "WA" else 0.84, size=n_samples)

        county_type = self.rng.choice(
            ["Metropolitan", "Micropolitan", "Rural"],
            size=n_samples,
            p=[0.6, 0.25, 0.15] if state == "WA" else [0.52, 0.28, 0.20],
        )
        health_service_area = self.rng.choice([f"HSA_{state}_{i:02d}" for i in range(1, 11)], size=n_samples)

        enrollment_month = self.rng.integers(1, 13, size=n_samples)
        seasonal_indicators = np.array([self._season_from_month(m) for m in enrollment_month])
        index_base = pd.Timestamp("2018-01-01") if state == "WA" else pd.Timestamp("2019-01-01")
        index_date = pd.Series(enrollment_month, dtype="int64").apply(
            lambda m: index_base + pd.offsets.MonthBegin(m)
        )

        logit = (
            -0.8
            + 0.04 * (age - 45)
            + 0.35 * diabetes
            + 0.42 * hypertension
            + 0.55 * heart_disease
            + 0.48 * mental_health
            + 0.50 * substance_use
            + 0.15 * (charlson_index - 3)
            + 0.18 * prior_ed
            + 0.35 * prior_hosp
            + 0.02 * medication_count
            + (0.15 if state == "VA" else 0)
        )
        probability = 1 / (1 + np.exp(-logit))
        acute_care = self.rng.binomial(1, np.clip(probability, 1e-3, 1 - 1e-3))

        df = pd.DataFrame(
            {
                "patient_id": [f"{state}_{i:06d}" for i in range(n_samples)],
                "state": state,
                "index_date": index_date.astype("datetime64[ns]"),
                "age": age,
                "gender": gender,
                "race_ethnicity": race_ethnicity,
                "urban_rural": urban_rural,
                "diabetes_mellitus": diabetes,
                "hypertension": hypertension,
                "heart_disease": heart_disease,
                "copd": copd,
                "mental_health_disorders": mental_health,
                "substance_abuse_disorders": substance_use,
                "charlson_comorbidity_index": charlson_index,
                "prior_ed_visits_12m": prior_ed,
                "prior_hospitalizations_12m": prior_hosp,
                "prior_outpatient_visits_12m": prior_outpatient,
                "medication_count": medication_count,
                "eligibility_category": eligibility,
                "enrollment_duration": enrollment_duration,
                "managed_care_enrollment": managed_care,
                "county_type": county_type,
                "health_service_area": health_service_area,
                "enrollment_month": enrollment_month,
                "seasonal_indicators": seasonal_indicators,
                "acute_care_utilization_12m": acute_care,
            }
        )

        return df

    @staticmethod
    def _season_from_month(month: int) -> str:
        if month in (12, 1, 2):
            return "Winter"
        if month in (3, 4, 5):
            return "Spring"
        if month in (6, 7, 8):
            return "Summer"
        return "Fall"

    def generate_medicaid_data(self, n_source: int, n_target: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate paired source and target domain datasets."""

        source_df = self.generate_state_population(n_source, state="WA")
        target_df = self.generate_state_population(n_target, state="VA")
        return source_df, target_df


class DataPreprocessor:
    """Prepare Medicaid datasets for downstream transfer learning experiments."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rng = np.random.default_rng(config.get("random_seed", 42))
        self.synthetic_generator = SyntheticDataGenerator(self.config.get("random_seed", 42))

    def load_and_preprocess(self) -> Tuple[DomainDataBundle, DomainDataBundle]:
        """Load raw data, apply preprocessing, and return structured datasets."""

        source_path = Path(self.config["data_paths"]["source_domain"])
        target_path = Path(self.config["data_paths"]["target_domain"])

        if not source_path.exists() or not target_path.exists():
            logger.warning("Medicaid data not found; generating synthetic datasets for local runs.")
            n_source = self.config.get("synthetic_defaults", {}).get("n_source", 6000)
            n_target = self.config.get("synthetic_defaults", {}).get("n_target", 4000)
            source_df, target_df = self.synthetic_generator.generate_medicaid_data(n_source, n_target)
            source_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            source_df.to_csv(source_path, index=False)
            target_df.to_csv(target_path, index=False)
        else:
            source_df = self._load_table(source_path)
            target_df = self._load_table(target_path)

        source_df = self._apply_cohort_criteria(source_df)
        target_df = self._apply_cohort_criteria(target_df)
        combined_df, feature_category_map = self._align_and_encode(source_df, target_df)

        source_bundle = self._split_domain_dataset(
            combined_df[combined_df["state"] == "WA"],
            feature_category_map,
            domain="source",
            raw_df=source_df,
        )
        target_bundle = self._split_domain_dataset(
            combined_df[combined_df["state"] == "VA"],
            feature_category_map,
            domain="target",
            raw_df=target_df,
        )

        self._log_domain_summary(source_bundle, target_bundle)
        return source_bundle, target_bundle

    def _load_table(self, path: Path) -> pd.DataFrame:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        raise ValueError(f"Unsupported file format: {path.suffix}")

    def _apply_cohort_criteria(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        inclusion = cfg.get("inclusion_criteria", {})
        exclusion = cfg.get("exclusion_criteria", {})

        original_size = len(df)
        df = df.copy()

        if "age_min" in inclusion and "age" in df.columns:
            df = df[df["age"] >= inclusion["age_min"]]
        if "age_max" in inclusion and "age" in df.columns:
            df = df[df["age"] <= inclusion["age_max"]]
        if inclusion.get("prior_acute_care_required") and "prior_ed_visits_12m" in df.columns:
            df = df[df["prior_ed_visits_12m"] + df.get("prior_hospitalizations_12m", 0) > 0]

        if exclusion.get("medicare_dual_eligible") and "medicare_dual_eligible" in df.columns:
            df = df[~df["medicare_dual_eligible"].astype(bool)]
        if exclusion.get("incomplete_demographics"):
            demo_cols = [col for col in ["gender", "race_ethnicity", "age"] if col in df.columns]
            df = df.dropna(subset=demo_cols)
        if exclusion.get("missing_outcome_data") and "acute_care_utilization_12m" in df.columns:
            df = df.dropna(subset=["acute_care_utilization_12m"])

        dropped = original_size - len(df)
        if dropped > 0:
            logger.info("Removed %s records based on cohort criteria", dropped)

        return df.reset_index(drop=True)

    def _align_and_encode(
        self, source_df: pd.DataFrame, target_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        features_cfg = self.config["features"]
        feature_cols = [col for cols in features_cfg.values() for col in cols if col in source_df.columns or col in target_df.columns]

        def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
            missing = set(feature_cols) - set(df.columns)
            for col in missing:
                df[col] = 0
            return df

        source_df = _ensure_columns(source_df).copy()
        target_df = _ensure_columns(target_df).copy()

        # Imputation statistics are computed on source_df only to avoid leakage
        # from the target domain into the feature representation used at test time.
        # String columns receive the sentinel value "Unknown"; numeric columns
        # receive the source-domain median.
        source_medians: Dict[str, float] = {}
        for col in source_df.columns:
            if source_df[col].dtype != "object":
                source_medians[col] = float(source_df[col].median())

        def _impute(df: pd.DataFrame) -> pd.DataFrame:
            for col in df.columns:
                if df[col].dtype == "object":
                    df[col] = df[col].fillna("Unknown")
                elif col in source_medians:
                    df[col] = df[col].fillna(source_medians[col])
                else:
                    df[col] = df[col].fillna(0.0)
            return df

        source_df = _impute(source_df)
        target_df = _impute(target_df)

        # Categorical encoding must use a fixed category set derived from
        # source_df only so that unseen target categories do not expand the
        # feature space.  The combined DataFrame is assembled only so that
        # get_dummies produces aligned columns; all category vocabulary comes
        # from source.
        categorical_cols: List[str] = []
        base_category_map: Dict[str, str] = {}
        for category_name, cols in features_cfg.items():
            for col in cols:
                if col not in source_df.columns:
                    continue
                base_category_map[col] = category_name
                if source_df[col].dtype == "object":
                    categorical_cols.append(col)

        # For categorical columns, restrict target values to those observed in
        # source training data.  Values not seen in source are mapped to "Unknown".
        source_categories: Dict[str, set] = {
            col: set(source_df[col].unique()) for col in categorical_cols
        }
        for col in categorical_cols:
            target_df[col] = target_df[col].apply(
                lambda v: v if v in source_categories[col] else "Unknown"
            )

        combined = pd.concat([source_df, target_df], axis=0, ignore_index=True)

        numeric_cols = [col for col in feature_cols if col not in categorical_cols]
        encoded_parts = []
        encoded_category_map: Dict[str, str] = {}

        if numeric_cols:
            encoded_parts.append(combined[numeric_cols].astype(float))
            for col in numeric_cols:
                encoded_category_map[col] = base_category_map.get(col, "other")

        for col in categorical_cols:
            one_hot = pd.get_dummies(combined[col], prefix=col, drop_first=self.config["preprocessing"]["categorical_encoding"].get("drop_first", True))
            encoded_parts.append(one_hot)
            for dummy_col in one_hot.columns:
                encoded_category_map[dummy_col] = base_category_map.get(col, "other")

        encoded_df = pd.concat(encoded_parts, axis=1)
        encoded_df["state"] = combined["state"].values
        if "index_date" in combined.columns:
            encoded_df["index_date"] = pd.to_datetime(combined["index_date"])
        encoded_df["outcome"] = combined["acute_care_utilization_12m"].astype(int).values

        return encoded_df, encoded_category_map

    def _split_domain_dataset(
        self,
        domain_df: pd.DataFrame,
        feature_category_map: Dict[str, str],
        domain: str,
        raw_df: Optional[pd.DataFrame] = None,
    ) -> DomainDataBundle:
        # Add a _row_id sentinel that carries through the sort so we can map
        # split indices back to raw_df positions for fairness/component analyses.
        domain_df = domain_df.copy()
        domain_df["_row_id"] = np.arange(len(domain_df))
        if "index_date" in domain_df.columns:
            domain_df = domain_df.sort_values("index_date").reset_index(drop=True)
        # _row_id now records the pre-sort position of each post-sort row.
        row_ids = domain_df["_row_id"].to_numpy()
        domain_df = domain_df.drop(columns=["_row_id"])
        feature_cols = [col for col in domain_df.columns if col not in {"state", "outcome", "index_date"}]
        X = domain_df[feature_cols].to_numpy(dtype=float)
        y = domain_df["outcome"].to_numpy(dtype=int)

        split_cfg = self.config["data_splitting"]["source_domain" if domain == "source" else "target_domain"]
        test_ratio = split_cfg.get("test_ratio", 0.2)
        val_ratio = split_cfg.get("validation_ratio", split_cfg.get("val_ratio", split_cfg.get("validation_split", 0.1)))

        all_indices = np.arange(len(X))

        if domain == "source":
            stratify = y if np.unique(y).size > 1 else None
            idx_train, idx_temp, _, _ = train_test_split(
                all_indices, y, test_size=test_ratio + val_ratio, stratify=stratify, random_state=42
            )
            X_train, X_temp = X[idx_train], X[idx_temp]
            y_train, y_temp = y[idx_train], y[idx_temp]
            val_share = val_ratio / (test_ratio + val_ratio) if (test_ratio + val_ratio) > 0 else 0.5
            idx_val, idx_test, _, _ = train_test_split(
                idx_temp, y_temp, test_size=1 - val_share,
                stratify=y_temp if stratify is not None else None, random_state=42
            )
            X_val, X_test = X[idx_val], X[idx_test]
            y_val, y_test = y[idx_val], y[idx_test]
            bundle = DomainDataBundle(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                X_full=X,
                y_full=y,
                feature_names=feature_cols,
                feature_category_map=feature_category_map,
                metadata={"domain": domain, "index_date": domain_df.get("index_date")},
            )
            return bundle

        # Target domain handling with support/query split
        stratify = y if np.unique(y).size > 1 else None
        idx_train, idx_temp, _, _ = train_test_split(
            all_indices, y, test_size=test_ratio + val_ratio, stratify=stratify, random_state=42
        )
        X_train, X_temp = X[idx_train], X[idx_temp]
        y_train, y_temp = y[idx_train], y[idx_temp]
        val_share = val_ratio / (test_ratio + val_ratio) if (test_ratio + val_ratio) > 0 else 0.5
        idx_val, idx_test, _, _ = train_test_split(
            idx_temp, y_temp, test_size=1 - val_share,
            stratify=y_temp if stratify is not None else None, random_state=42
        )
        X_val, X_test = X[idx_val], X[idx_test]
        y_val, y_test = y[idx_val], y[idx_test]

        support_ratio = split_cfg.get("support_ratio", 0.1)
        support_size = max(1, int(len(X_train) * support_ratio))
        support_indices = self.rng.choice(len(X_train), size=support_size, replace=False)
        query_indices = np.setdiff1d(np.arange(len(X_train)), support_indices)

        support_X = X_train[support_indices]
        support_y = y_train[support_indices]
        if query_indices.size == 0:
            query_X = support_X
            query_y = support_y
        else:
            query_X = X_train[query_indices]
            query_y = y_train[query_indices]

        # Build test-set demographic DataFrame (race_ethnicity, gender, age) from
        # raw_df. raw_df was reset_index after cohort filtering; row_ids[idx_test]
        # maps sorted positions back to the original raw_df row positions.
        raw_test_ids = row_ids[idx_test]
        test_demographics: Optional[pd.DataFrame] = None
        if raw_df is not None:
            demo_cols = [c for c in ["race_ethnicity", "gender", "age"] if c in raw_df.columns]
            if demo_cols:
                test_demographics = raw_df[demo_cols].iloc[raw_test_ids].reset_index(drop=True)

        bundle = DomainDataBundle(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            X_full=X,
            y_full=y,
            feature_names=feature_cols,
            feature_category_map=feature_category_map,
            metadata={"domain": domain, "index_date": domain_df.get("index_date")},
            support_X=support_X,
            support_y=support_y,
            query_X=query_X,
            query_y=query_y,
            test_demographics=test_demographics,
            raw_df=raw_df,
            test_indices=raw_test_ids,
        )
        return bundle

    def _log_domain_summary(self, source_bundle: DomainDataBundle, target_bundle: DomainDataBundle) -> None:
        logger.info("Source domain size: train=%s, val=%s, test=%s", len(source_bundle.X_train), len(source_bundle.X_val), len(source_bundle.X_test))
        logger.info("Target domain size: train=%s, support=%s, test=%s", len(target_bundle.X_train), len(target_bundle.support_X), len(target_bundle.X_test))
        logger.info("Outcome prevalence - source train: %.3f, target test: %.3f", source_bundle.y_train.mean(), target_bundle.y_test.mean())
