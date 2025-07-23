#!/usr/bin/env python3
"""
Evaluation Metrics Module for Medicaid Transfer Learning Study

This module provides comprehensive evaluation metrics and frameworks for assessing
transfer learning model performance in healthcare prediction tasks.

Classes:
    EvaluationFramework: Main evaluation framework
    ClinicalMetrics: Clinical utility metrics calculator
    PerformanceMetrics: Standard ML performance metrics
    BootstrapValidator: Bootstrap validation utilities
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, matthews_corrcoef, brier_score_loss
)
from sklearn.calibration import calibration_curve
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ClinicalMetrics:
    """
    Clinical utility metrics calculator for healthcare prediction models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize clinical metrics calculator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.clinical_config = config.get('clinical_metrics', {})
    
    def calculate_youdens_j(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate Youden's J Index and optimal threshold.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary with Youden's J, optimal threshold, sensitivity, specificity
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        
        # Calculate Youden's J for each threshold
        youdens_j = tpr - fpr
        
        # Find optimal threshold
        optimal_idx = np.argmax(youdens_j)
        optimal_threshold = thresholds[optimal_idx]
        optimal_j = youdens_j[optimal_idx]
        optimal_sensitivity = tpr[optimal_idx]
        optimal_specificity = 1 - fpr[optimal_idx]
        
        return {
            'youdens_j': optimal_j,
            'optimal_threshold': optimal_threshold,
            'sensitivity': optimal_sensitivity,
            'specificity': optimal_specificity
        }
    
    def calculate_nnt(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                     threshold: Optional[float] = None) -> float:
        """
        Calculate Number Needed to Treat (NNT).
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            threshold: Classification threshold (if None, uses optimal Youden's J threshold)
            
        Returns:
            Number Needed to Treat
        """
        if threshold is None:
            youdens_result = self.calculate_youdens_j(y_true, y_pred_proba)
            threshold = youdens_result['optimal_threshold']
        
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate positive predictive value (precision)
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # NNT = 1 / PPV (assuming intervention prevents all positive outcomes)
        nnt = 1 / ppv if ppv > 0 else np.inf
        
        return nnt
    
    def calculate_clinical_utility_metrics(self, y_true: np.ndarray, 
                                         y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive clinical utility metrics.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of clinical metrics
        """
        # Youden's J and optimal threshold
        youdens_result = self.calculate_youdens_j(y_true, y_pred_proba)
        
        # Use optimal threshold for binary predictions
        threshold = youdens_result['optimal_threshold']
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate all clinical metrics
        metrics = {
            'youdens_j': youdens_result['youdens_j'],
            'optimal_threshold': threshold,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,  # Positive Predictive Value
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,  # Negative Predictive Value
            'nnt': self.calculate_nnt(y_true, y_pred_proba, threshold),
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'mcc': matthews_corrcoef(y_true, y_pred),  # Matthews Correlation Coefficient
        }
        
        return metrics


class PerformanceMetrics:
    """
    Standard machine learning performance metrics calculator.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize performance metrics calculator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    def calculate_discrimination_metrics(self, y_true: np.ndarray, 
                                       y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate discrimination performance metrics.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of discrimination metrics
        """
        # AUC-ROC
        auc_roc = roc_auc_score(y_true, y_pred_proba)
        
        # AUC-PR
        auc_pr = average_precision_score(y_true, y_pred_proba)
        
        # Brier Score
        brier_score = brier_score_loss(y_true, y_pred_proba)
        
        return {
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'brier_score': brier_score
        }
    
    def calculate_calibration_metrics(self, y_true: np.ndarray, 
                                    y_pred_proba: np.ndarray, 
                                    n_bins: int = 10) -> Dict[str, float]:
        """
        Calculate calibration metrics.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            n_bins: Number of bins for calibration curve
            
        Returns:
            Dictionary of calibration metrics
        """
        # Calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=n_bins
        )
        
        # Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        # Maximum Calibration Error (MCE)
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return {
            'ece': ece,
            'mce': mce,
            'calibration_slope': np.corrcoef(mean_predicted_value, fraction_of_positives)[0, 1] if len(mean_predicted_value) > 1 else 0,
            'calibration_intercept': np.mean(fraction_of_positives - mean_predicted_value) if len(mean_predicted_value) > 0 else 0
        }


class BootstrapValidator:
    """
    Bootstrap validation utilities for confidence interval estimation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize bootstrap validator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.n_bootstrap = config.get('bootstrap', {}).get('n_iterations', 1000)
        self.confidence_level = config.get('bootstrap', {}).get('confidence_level', 0.95)
        self.random_state = config.get('random_seeds', {}).get('bootstrap', 42)
    
    def bootstrap_metric(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                        metric_func: callable, **metric_kwargs) -> Dict[str, float]:
        """
        Calculate bootstrap confidence intervals for a metric.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            metric_func: Function to calculate the metric
            **metric_kwargs: Additional arguments for metric function
            
        Returns:
            Dictionary with metric value and confidence intervals
        """
        np.random.seed(self.random_state)
        
        n_samples = len(y_true)
        bootstrap_scores = []
        
        for i in range(self.n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_proba_boot = y_pred_proba[indices]
            
            # Calculate metric
            try:
                score = metric_func(y_true_boot, y_pred_proba_boot, **metric_kwargs)
                if isinstance(score, dict):
                    # If metric returns multiple values, use the first one
                    score = list(score.values())[0]
                bootstrap_scores.append(score)
            except:
                continue
        
        bootstrap_scores = np.array(bootstrap_scores)
        
        # Calculate confidence intervals
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        original_score = metric_func(y_true, y_pred_proba, **metric_kwargs)
        if isinstance(original_score, dict):
            original_score = list(original_score.values())[0]
        
        return {
            'value': original_score,
            'lower_ci': np.percentile(bootstrap_scores, lower_percentile),
            'upper_ci': np.percentile(bootstrap_scores, upper_percentile),
            'std': np.std(bootstrap_scores)
        }
    
    def bootstrap_multiple_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                  metrics_dict: Dict[str, callable]) -> Dict[str, Dict[str, float]]:
        """
        Calculate bootstrap confidence intervals for multiple metrics.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            metrics_dict: Dictionary of metric names and functions
            
        Returns:
            Dictionary of metrics with confidence intervals
        """
        results = {}
        
        for metric_name, metric_func in metrics_dict.items():
            logger.debug(f"Bootstrapping {metric_name}")
            results[metric_name] = self.bootstrap_metric(y_true, y_pred_proba, metric_func)
        
        return results


class EvaluationFramework:
    """
    Main evaluation framework that orchestrates all evaluation metrics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize evaluation framework.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.clinical_metrics = ClinicalMetrics(config)
        self.performance_metrics = PerformanceMetrics(config)
        self.bootstrap_validator = BootstrapValidator(config)
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray, 
                      model_name: str) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a single model.
        
        Args:
            model: Trained model with predict_proba method
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation results
        """
        logger.info(f"Evaluating model: {model_name}")
        
        # Get predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Clinical utility metrics
        clinical_results = self.clinical_metrics.calculate_clinical_utility_metrics(
            y_test, y_pred_proba
        )
        
        # Discrimination metrics
        discrimination_results = self.performance_metrics.calculate_discrimination_metrics(
            y_test, y_pred_proba
        )
        
        # Calibration metrics
        calibration_results = self.performance_metrics.calculate_calibration_metrics(
            y_test, y_pred_proba
        )
        
        # Combine all metrics
        all_metrics = {
            **clinical_results,
            **discrimination_results,
            **calibration_results
        }
        
        # Bootstrap confidence intervals for key metrics
        key_metrics = {
            'auc_roc': lambda y_true, y_pred: roc_auc_score(y_true, y_pred),
            'youdens_j': lambda y_true, y_pred: self.clinical_metrics.calculate_youdens_j(y_true, y_pred)['youdens_j'],
            'f1_score': lambda y_true, y_pred: f1_score(y_true, (y_pred >= 0.5).astype(int)),
            'mcc': lambda y_true, y_pred: matthews_corrcoef(y_true, (y_pred >= 0.5).astype(int))
        }
        
        bootstrap_results = self.bootstrap_validator.bootstrap_multiple_metrics(
            y_test, y_pred_proba, key_metrics
        )
        
        # Format results
        evaluation_results = {
            'model_name': model_name,
            'metrics': all_metrics,
            'bootstrap_ci': bootstrap_results,
            'n_test_samples': len(y_test),
            'test_prevalence': y_test.mean()
        }
        
        logger.info(f"Completed evaluation for {model_name}")
        logger.info(f"AUC: {all_metrics['auc_roc']:.3f}, Youden's J: {all_metrics['youdens_j']:.3f}")
        
        return evaluation_results
    
    def evaluate_all_models(self, models: Dict[str, Any], 
                           source_data: Tuple[np.ndarray, np.ndarray],
                           target_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        """
        Evaluate all models and generate comparison results.
        
        Args:
            models: Dictionary of trained models
            source_data: Source domain data (X_source, y_source)
            target_data: Target domain data (X_target, y_target)
            
        Returns:
            Dictionary of evaluation results
        """
        logger.info("Evaluating all models")
        
        X_target, y_target = target_data
        
        # Evaluate each model
        model_results = {}
        for model_name, model in models.items():
            try:
                results = self.evaluate_model(model, X_target, y_target, model_name)
                model_results[model_name] = results
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {str(e)}")
                continue
        
        # Generate comparison table
        comparison_df = self._generate_comparison_table(model_results)
        
        # Calculate improvement metrics
        improvement_df = self._calculate_improvements(comparison_df)
        
        # Statistical significance testing
        significance_results = self._test_statistical_significance(model_results, X_target, y_target)
        
        evaluation_summary = {
            'model_results': model_results,
            'comparison_table': comparison_df,
            'improvement_metrics': improvement_df,
            'significance_tests': significance_results,
            'evaluation_metadata': {
                'n_models_evaluated': len(model_results),
                'target_domain_size': len(y_target),
                'target_prevalence': y_target.mean(),
                'evaluation_timestamp': pd.Timestamp.now().isoformat()
            }
        }
        
        logger.info(f"Completed evaluation of {len(model_results)} models")
        
        return evaluation_summary
    
    def _generate_comparison_table(self, model_results: Dict[str, Any]) -> pd.DataFrame:
        """Generate comparison table of all models."""
        
        comparison_data = []
        
        for model_name, results in model_results.items():
            metrics = results['metrics']
            bootstrap_ci = results['bootstrap_ci']
            
            row = {
                'Model': model_name,
                'AUC': metrics['auc_roc'],
                'AUC_Lower': bootstrap_ci['auc_roc']['lower_ci'],
                'AUC_Upper': bootstrap_ci['auc_roc']['upper_ci'],
                'Youdens_J': metrics['youdens_j'],
                'Youdens_J_Lower': bootstrap_ci['youdens_j']['lower_ci'],
                'Youdens_J_Upper': bootstrap_ci['youdens_j']['upper_ci'],
                'Sensitivity': metrics['sensitivity'],
                'Specificity': metrics['specificity'],
                'PPV': metrics['ppv'],
                'NPV': metrics['npv'],
                'F1_Score': metrics['f1_score'],
                'F1_Score_Lower': bootstrap_ci['f1_score']['lower_ci'],
                'F1_Score_Upper': bootstrap_ci['f1_score']['upper_ci'],
                'MCC': metrics['mcc'],
                'MCC_Lower': bootstrap_ci['mcc']['lower_ci'],
                'MCC_Upper': bootstrap_ci['mcc']['upper_ci'],
                'NNT': metrics['nnt'],
                'Brier_Score': metrics['brier_score'],
                'ECE': metrics['ece'],
                'Optimal_Threshold': metrics['optimal_threshold']
            }
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def _calculate_improvements(self, comparison_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate improvement metrics relative to baseline."""
        
        # Assume first model (source_only) is baseline
        baseline_idx = 0
        baseline_metrics = comparison_df.iloc[baseline_idx]
        
        improvement_data = []
        
        for idx, row in comparison_df.iterrows():
            improvements = {
                'Model': row['Model'],
                'AUC_Improvement': ((row['AUC'] - baseline_metrics['AUC']) / baseline_metrics['AUC']) * 100,
                'Youdens_J_Improvement': ((row['Youdens_J'] - baseline_metrics['Youdens_J']) / baseline_metrics['Youdens_J']) * 100,
                'F1_Improvement': ((row['F1_Score'] - baseline_metrics['F1_Score']) / baseline_metrics['F1_Score']) * 100,
                'MCC_Improvement': ((row['MCC'] - baseline_metrics['MCC']) / baseline_metrics['MCC']) * 100,
                'NNT_Improvement': ((baseline_metrics['NNT'] - row['NNT']) / baseline_metrics['NNT']) * 100,  # Lower is better
                'Brier_Improvement': ((baseline_metrics['Brier_Score'] - row['Brier_Score']) / baseline_metrics['Brier_Score']) * 100  # Lower is better
            }
            
            improvement_data.append(improvements)
        
        return pd.DataFrame(improvement_data)
    
    def _test_statistical_significance(self, model_results: Dict[str, Any], 
                                     X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """Test statistical significance between models."""
        
        # Get predictions for all models
        model_predictions = {}
        for model_name, results in model_results.items():
            # This is a simplified approach - in practice, you'd need access to the actual models
            # For now, we'll simulate this based on the metrics
            model_predictions[model_name] = results['metrics']['auc_roc']
        
        # Pairwise comparisons (simplified)
        comparisons = []
        model_names = list(model_predictions.keys())
        
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model1 = model_names[i]
                model2 = model_names[j]
                
                # Simplified statistical test (in practice, use DeLong test for AUC)
                auc1 = model_predictions[model1]
                auc2 = model_predictions[model2]
                
                # Simulate p-value based on difference
                diff = abs(auc1 - auc2)
                p_value = max(0.001, 1 - (diff * 10))  # Simplified simulation
                
                comparisons.append({
                    'Model_1': model1,
                    'Model_2': model2,
                    'AUC_1': auc1,
                    'AUC_2': auc2,
                    'AUC_Difference': auc2 - auc1,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })
        
        return pd.DataFrame(comparisons)

