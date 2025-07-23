#!/usr/bin/env python3
"""
Data Preprocessing Module for Medicaid Transfer Learning Study

This module handles data loading, preprocessing, quality checks, and feature engineering
for the cross-state Medicaid transfer learning analysis.

Classes:
    DataPreprocessor: Main class for data preprocessing operations
    FeatureEngineer: Feature engineering and selection utilities
    DataQualityChecker: Data quality assessment and validation
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DataQualityChecker:
    """
    Data quality assessment and validation utilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data quality checker.
        
        Args:
            config: Data configuration dictionary
        """
        self.config = config
        self.quality_thresholds = config.get('quality_thresholds', {})
    
    def check_data_quality(self, df: pd.DataFrame, domain_name: str) -> Dict[str, Any]:
        """
        Perform comprehensive data quality checks.
        
        Args:
            df: Input dataframe
            domain_name: Name of the domain (source/target)
            
        Returns:
            Dictionary of quality metrics
        """
        logger.info(f"Performing data quality checks for {domain_name} domain")
        
        quality_report = {
            'domain': domain_name,
            'total_records': len(df),
            'total_features': len(df.columns),
            'missing_data': {},
            'data_types': {},
            'outliers': {},
            'duplicates': 0,
            'quality_score': 0.0
        }
        
        # Missing data analysis
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        quality_report['missing_data'] = {
            'features_with_missing': (missing_counts > 0).sum(),
            'max_missing_percentage': missing_percentages.max(),
            'features_above_threshold': (missing_percentages > 
                                       self.quality_thresholds.get('missing_threshold', 20)).sum()
        }
        
        # Data type analysis
        quality_report['data_types'] = {
            'numeric_features': df.select_dtypes(include=[np.number]).shape[1],
            'categorical_features': df.select_dtypes(include=['object']).shape[1],
            'datetime_features': df.select_dtypes(include=['datetime']).shape[1]
        }
        
        # Duplicate records
        quality_report['duplicates'] = df.duplicated().sum()
        
        # Outlier detection for numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_counts = {}
        
        for col in numeric_cols:
            if df[col].notna().sum() > 0:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outlier_counts[col] = outliers
        
        quality_report['outliers'] = {
            'features_with_outliers': len([k for k, v in outlier_counts.items() if v > 0]),
            'max_outlier_count': max(outlier_counts.values()) if outlier_counts else 0,
            'total_outliers': sum(outlier_counts.values())
        }
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(quality_report)
        quality_report['quality_score'] = quality_score
        
        logger.info(f"{domain_name} domain quality score: {quality_score:.2f}/100")
        
        return quality_report
    
    def _calculate_quality_score(self, quality_report: Dict[str, Any]) -> float:
        """
        Calculate overall data quality score (0-100).
        
        Args:
            quality_report: Quality metrics dictionary
            
        Returns:
            Quality score between 0 and 100
        """
        score = 100.0
        
        # Penalize for missing data
        max_missing = quality_report['missing_data']['max_missing_percentage']
        if max_missing > 50:
            score -= 30
        elif max_missing > 20:
            score -= 15
        elif max_missing > 10:
            score -= 5
        
        # Penalize for duplicates
        duplicate_rate = quality_report['duplicates'] / quality_report['total_records']
        if duplicate_rate > 0.1:
            score -= 20
        elif duplicate_rate > 0.05:
            score -= 10
        
        # Penalize for excessive outliers
        outlier_rate = quality_report['outliers']['total_outliers'] / quality_report['total_records']
        if outlier_rate > 0.2:
            score -= 15
        elif outlier_rate > 0.1:
            score -= 8
        
        return max(0.0, score)


class FeatureEngineer:
    """
    Feature engineering and selection utilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature engineer.
        
        Args:
            config: Data configuration dictionary
        """
        self.config = config
        self.feature_config = config.get('feature_engineering', {})
        
        # Feature categories
        self.demographic_features = []
        self.clinical_features = []
        self.utilization_features = []
        self.temporal_features = []
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform comprehensive feature engineering.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with engineered features
        """
        logger.info("Starting feature engineering")
        
        df_engineered = df.copy()
        
        # Demographic features
        df_engineered = self._engineer_demographic_features(df_engineered)
        
        # Clinical features
        df_engineered = self._engineer_clinical_features(df_engineered)
        
        # Healthcare utilization features
        df_engineered = self._engineer_utilization_features(df_engineered)
        
        # Temporal features
        df_engineered = self._engineer_temporal_features(df_engineered)
        
        # Interaction features
        df_engineered = self._engineer_interaction_features(df_engineered)
        
        logger.info(f"Feature engineering completed. Features: {len(df_engineered.columns)}")
        
        return df_engineered
    
    def _engineer_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer demographic features."""
        
        # Age groups
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'], 
                                   bins=[0, 18, 35, 50, 65, 100], 
                                   labels=['0-17', '18-34', '35-49', '50-64', '65+'])
            df['age_squared'] = df['age'] ** 2
            df['is_elderly'] = (df['age'] >= 65).astype(int)
        
        # Gender interactions
        if 'gender' in df.columns and 'age' in df.columns:
            df['female_elderly'] = ((df['gender'] == 'F') & (df['age'] >= 65)).astype(int)
        
        # Race/ethnicity features
        if 'race' in df.columns:
            # Create binary indicators for major racial groups
            race_categories = ['White', 'Black', 'Hispanic', 'Asian', 'Other']
            for category in race_categories:
                df[f'race_{category.lower()}'] = (df['race'] == category).astype(int)
        
        return df
    
    def _engineer_clinical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer clinical features."""
        
        # Charlson Comorbidity Index components
        charlson_conditions = [
            'myocardial_infarction', 'congestive_heart_failure', 'peripheral_vascular_disease',
            'cerebrovascular_disease', 'dementia', 'chronic_pulmonary_disease',
            'rheumatic_disease', 'peptic_ulcer_disease', 'mild_liver_disease',
            'diabetes_without_complications', 'diabetes_with_complications',
            'hemiplegia_or_paraplegia', 'renal_disease', 'malignancy',
            'moderate_severe_liver_disease', 'metastatic_solid_tumor', 'aids'
        ]
        
        # Calculate Charlson score if components are available
        available_conditions = [col for col in charlson_conditions if col in df.columns]
        if available_conditions:
            df['charlson_score'] = df[available_conditions].sum(axis=1)
            df['has_comorbidity'] = (df['charlson_score'] > 0).astype(int)
            df['multiple_comorbidities'] = (df['charlson_score'] > 2).astype(int)
        
        # Mental health indicators
        mental_health_conditions = ['depression', 'anxiety', 'bipolar', 'schizophrenia']
        available_mh = [col for col in mental_health_conditions if col in df.columns]
        if available_mh:
            df['mental_health_count'] = df[available_mh].sum(axis=1)
            df['has_mental_health'] = (df['mental_health_count'] > 0).astype(int)
        
        # Substance use indicators
        substance_conditions = ['alcohol_abuse', 'drug_abuse', 'tobacco_use']
        available_substance = [col for col in substance_conditions if col in df.columns]
        if available_substance:
            df['substance_use_count'] = df[available_substance].sum(axis=1)
            df['has_substance_use'] = (df['substance_use_count'] > 0).astype(int)
        
        return df
    
    def _engineer_utilization_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer healthcare utilization features."""
        
        # Prior utilization patterns
        utilization_cols = ['prior_ed_visits', 'prior_hospitalizations', 'prior_outpatient_visits']
        available_util = [col for col in utilization_cols if col in df.columns]
        
        if available_util:
            # Total prior utilization
            df['total_prior_utilization'] = df[available_util].sum(axis=1)
            
            # High utilizer indicators
            if 'prior_ed_visits' in df.columns:
                df['high_ed_utilizer'] = (df['prior_ed_visits'] >= 4).astype(int)
            
            if 'prior_hospitalizations' in df.columns:
                df['frequent_hospitalizations'] = (df['prior_hospitalizations'] >= 2).astype(int)
            
            # Utilization ratios
            if 'prior_ed_visits' in df.columns and 'prior_outpatient_visits' in df.columns:
                df['ed_to_outpatient_ratio'] = (df['prior_ed_visits'] / 
                                               (df['prior_outpatient_visits'] + 1))
        
        # Medication-related features
        if 'medication_count' in df.columns:
            df['polypharmacy'] = (df['medication_count'] >= 5).astype(int)
            df['medication_complexity'] = pd.cut(df['medication_count'],
                                                bins=[0, 2, 5, 10, 100],
                                                labels=['Low', 'Medium', 'High', 'Very High'])
        
        return df
    
    def _engineer_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer temporal features."""
        
        # Enrollment duration
        if 'enrollment_start_date' in df.columns and 'enrollment_end_date' in df.columns:
            df['enrollment_duration'] = (pd.to_datetime(df['enrollment_end_date']) - 
                                        pd.to_datetime(df['enrollment_start_date'])).dt.days
            df['long_term_enrollee'] = (df['enrollment_duration'] >= 365).astype(int)
        
        # Seasonality features
        if 'index_date' in df.columns:
            df['index_date'] = pd.to_datetime(df['index_date'])
            df['index_month'] = df['index_date'].dt.month
            df['index_quarter'] = df['index_date'].dt.quarter
            df['index_season'] = df['index_month'].map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Fall', 10: 'Fall', 11: 'Fall'
            })
        
        return df
    
    def _engineer_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer interaction features."""
        
        # Age-comorbidity interactions
        if 'age' in df.columns and 'charlson_score' in df.columns:
            df['age_charlson_interaction'] = df['age'] * df['charlson_score']
        
        # Gender-utilization interactions
        if 'gender' in df.columns and 'total_prior_utilization' in df.columns:
            df['female_high_utilizer'] = ((df['gender'] == 'F') & 
                                         (df['total_prior_utilization'] > df['total_prior_utilization'].median())).astype(int)
        
        return df


class DataPreprocessor:
    """
    Main data preprocessing class that orchestrates all preprocessing operations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data preprocessor.
        
        Args:
            config: Data configuration dictionary
        """
        self.config = config
        self.quality_checker = DataQualityChecker(config)
        self.feature_engineer = FeatureEngineer(config)
        
        # Preprocessing components
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoders = {}
        
    def load_and_preprocess(self) -> Tuple[Tuple[np.ndarray, np.ndarray], 
                                          Tuple[np.ndarray, np.ndarray]]:
        """
        Load and preprocess both source and target domain data.
        
        Returns:
            Tuple of ((X_source, y_source), (X_target, y_target))
        """
        logger.info("Starting data loading and preprocessing")
        
        # Load raw data
        source_df = self._load_domain_data('source')
        target_df = self._load_domain_data('target')
        
        # Data quality checks
        source_quality = self.quality_checker.check_data_quality(source_df, 'source')
        target_quality = self.quality_checker.check_data_quality(target_df, 'target')
        
        # Feature engineering
        source_df = self.feature_engineer.engineer_features(source_df)
        target_df = self.feature_engineer.engineer_features(target_df)
        
        # Preprocessing
        source_data = self._preprocess_domain_data(source_df, 'source')
        target_data = self._preprocess_domain_data(target_df, 'target')
        
        logger.info("Data loading and preprocessing completed")
        
        return source_data, target_data
    
    def _load_domain_data(self, domain: str) -> pd.DataFrame:
        """
        Load data for a specific domain.
        
        Args:
            domain: Domain name ('source' or 'target')
            
        Returns:
            Loaded dataframe
        """
        data_path = self.config['data_paths'][domain]
        
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        else:
            # Generate synthetic data for demonstration
            df = self._generate_synthetic_data(domain)
        
        logger.info(f"Loaded {domain} domain data: {df.shape}")
        return df
    
    def _generate_synthetic_data(self, domain: str) -> pd.DataFrame:
        """
        Generate synthetic Medicaid data for demonstration purposes.
        
        Args:
            domain: Domain name ('source' or 'target')
            
        Returns:
            Synthetic dataframe
        """
        np.random.seed(42 if domain == 'source' else 123)
        
        # Sample sizes
        n_samples = 50000 if domain == 'source' else 60000
        
        # Generate synthetic features
        data = {
            # Demographics
            'age': np.random.normal(45, 15, n_samples).clip(18, 85),
            'gender': np.random.choice(['M', 'F'], n_samples, p=[0.4, 0.6]),
            'race': np.random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Other'], 
                                   n_samples, p=[0.5, 0.25, 0.15, 0.05, 0.05]),
            
            # Clinical conditions (Charlson components)
            'myocardial_infarction': np.random.binomial(1, 0.05, n_samples),
            'congestive_heart_failure': np.random.binomial(1, 0.08, n_samples),
            'peripheral_vascular_disease': np.random.binomial(1, 0.06, n_samples),
            'cerebrovascular_disease': np.random.binomial(1, 0.04, n_samples),
            'dementia': np.random.binomial(1, 0.03, n_samples),
            'chronic_pulmonary_disease': np.random.binomial(1, 0.12, n_samples),
            'rheumatic_disease': np.random.binomial(1, 0.02, n_samples),
            'peptic_ulcer_disease': np.random.binomial(1, 0.01, n_samples),
            'mild_liver_disease': np.random.binomial(1, 0.03, n_samples),
            'diabetes_without_complications': np.random.binomial(1, 0.15, n_samples),
            'diabetes_with_complications': np.random.binomial(1, 0.05, n_samples),
            'hemiplegia_or_paraplegia': np.random.binomial(1, 0.01, n_samples),
            'renal_disease': np.random.binomial(1, 0.07, n_samples),
            'malignancy': np.random.binomial(1, 0.04, n_samples),
            'moderate_severe_liver_disease': np.random.binomial(1, 0.01, n_samples),
            'metastatic_solid_tumor': np.random.binomial(1, 0.01, n_samples),
            'aids': np.random.binomial(1, 0.005, n_samples),
            
            # Mental health
            'depression': np.random.binomial(1, 0.20, n_samples),
            'anxiety': np.random.binomial(1, 0.15, n_samples),
            'bipolar': np.random.binomial(1, 0.03, n_samples),
            'schizophrenia': np.random.binomial(1, 0.01, n_samples),
            
            # Substance use
            'alcohol_abuse': np.random.binomial(1, 0.08, n_samples),
            'drug_abuse': np.random.binomial(1, 0.05, n_samples),
            'tobacco_use': np.random.binomial(1, 0.25, n_samples),
            
            # Healthcare utilization
            'prior_ed_visits': np.random.poisson(1.2, n_samples),
            'prior_hospitalizations': np.random.poisson(0.3, n_samples),
            'prior_outpatient_visits': np.random.poisson(8, n_samples),
            'medication_count': np.random.poisson(3, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Generate outcome with realistic relationships
        # Higher risk for older patients, more comorbidities, higher utilization
        risk_score = (
            0.02 * (df['age'] - 45) +
            0.3 * df['congestive_heart_failure'] +
            0.2 * df['diabetes_with_complications'] +
            0.15 * df['renal_disease'] +
            0.1 * df['prior_ed_visits'] +
            0.2 * df['prior_hospitalizations'] +
            np.random.normal(0, 0.5, n_samples)
        )
        
        # Domain shift: target domain has higher baseline risk
        if domain == 'target':
            risk_score += 0.3
        
        # Convert to probabilities and generate binary outcome
        probabilities = 1 / (1 + np.exp(-risk_score))
        df['acute_care_utilization'] = np.random.binomial(1, probabilities, n_samples)
        
        logger.info(f"Generated synthetic {domain} data with {len(df)} samples")
        logger.info(f"Outcome prevalence: {df['acute_care_utilization'].mean():.3f}")
        
        return df
    
    def _preprocess_domain_data(self, df: pd.DataFrame, domain: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data for a specific domain.
        
        Args:
            df: Input dataframe
            domain: Domain name ('source' or 'target')
            
        Returns:
            Tuple of (X, y) arrays
        """
        logger.info(f"Preprocessing {domain} domain data")
        
        # Separate features and target
        target_col = self.config['target_column']
        y = df[target_col].values
        X_df = df.drop(columns=[target_col])
        
        # Handle categorical variables
        categorical_cols = X_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X_df[col] = self.label_encoders[col].fit_transform(X_df[col].astype(str))
            else:
                # Handle unseen categories
                X_df[col] = X_df[col].astype(str)
                known_classes = set(self.label_encoders[col].classes_)
                X_df[col] = X_df[col].apply(lambda x: x if x in known_classes else 'unknown')
                
                # Add 'unknown' to encoder if not present
                if 'unknown' not in known_classes:
                    self.label_encoders[col].classes_ = np.append(
                        self.label_encoders[col].classes_, 'unknown'
                    )
                
                X_df[col] = self.label_encoders[col].transform(X_df[col])
        
        # Handle missing values
        X = X_df.values
        if domain == 'source':
            X = self.imputer.fit_transform(X)
        else:
            X = self.imputer.transform(X)
        
        logger.info(f"Preprocessed {domain} data: X shape {X.shape}, y shape {y.shape}")
        logger.info(f"Outcome prevalence: {y.mean():.3f}")
        
        return X, y

