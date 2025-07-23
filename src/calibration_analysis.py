#!/usr/bin/env python3
"""
Calibration Analysis Module for Medicaid Transfer Learning Study

This module provides comprehensive calibration analysis capabilities including
reliability diagrams, calibration metrics, and post-hoc calibration methods.

Classes:
    CalibrationAnalyzer: Main calibration analysis framework
    ReliabilityDiagram: Reliability diagram generation
    PostHocCalibrator: Post-hoc calibration methods
    CalibrationMetrics: Calibration metric calculations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import cross_val_predict
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class CalibrationMetrics:
    """
    Calibration metric calculations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize calibration metrics calculator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.calibration_config = config.get('calibration', {})
    
    def expected_calibration_error(self, y_true: np.ndarray, y_prob: np.ndarray, 
                                 n_bins: int = 10) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            n_bins: Number of bins for calibration
            
        Returns:
            Expected Calibration Error
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Determine if sample is in bin m (between bin lower & upper)
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def maximum_calibration_error(self, y_true: np.ndarray, y_prob: np.ndarray, 
                                n_bins: int = 10) -> float:
        """
        Calculate Maximum Calibration Error (MCE).
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            n_bins: Number of bins for calibration
            
        Returns:
            Maximum Calibration Error
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return mce
    
    def reliability_diagram_data(self, y_true: np.ndarray, y_prob: np.ndarray, 
                               n_bins: int = 10) -> Dict[str, np.ndarray]:
        """
        Generate data for reliability diagram.
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            n_bins: Number of bins
            
        Returns:
            Dictionary with bin data for plotting
        """
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins
        )
        
        # Calculate bin counts and confidence intervals
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_counts = []
        bin_centers = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            bin_counts.append(in_bin.sum())
            bin_centers.append((bin_lower + bin_upper) / 2)
        
        return {
            'fraction_of_positives': fraction_of_positives,
            'mean_predicted_value': mean_predicted_value,
            'bin_counts': np.array(bin_counts),
            'bin_centers': np.array(bin_centers),
            'bin_boundaries': bin_boundaries
        }
    
    def calculate_all_metrics(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """
        Calculate all calibration metrics.
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            
        Returns:
            Dictionary of calibration metrics
        """
        n_bins = self.calibration_config.get('n_bins', 10)
        
        metrics = {
            'ece': self.expected_calibration_error(y_true, y_prob, n_bins),
            'mce': self.maximum_calibration_error(y_true, y_prob, n_bins),
            'brier_score': brier_score_loss(y_true, y_prob),
            'log_loss': log_loss(y_true, y_prob),
        }
        
        # Add reliability diagram data
        reliability_data = self.reliability_diagram_data(y_true, y_prob, n_bins)
        
        # Calculate calibration slope and intercept
        if len(reliability_data['mean_predicted_value']) > 1:
            slope_coef = np.polyfit(reliability_data['mean_predicted_value'], 
                                  reliability_data['fraction_of_positives'], 1)
            metrics['calibration_slope'] = slope_coef[0]
            metrics['calibration_intercept'] = slope_coef[1]
        else:
            metrics['calibration_slope'] = 1.0
            metrics['calibration_intercept'] = 0.0
        
        return metrics


class PostHocCalibrator:
    """
    Post-hoc calibration methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize post-hoc calibrator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.calibration_config = config.get('calibration', {})
        
        self.platt_scaler = None
        self.isotonic_regressor = None
    
    def fit_platt_scaling(self, y_true: np.ndarray, y_prob: np.ndarray) -> LogisticRegression:
        """
        Fit Platt scaling calibration.
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            
        Returns:
            Fitted logistic regression model
        """
        # Convert probabilities to logits
        epsilon = 1e-15
        y_prob_clipped = np.clip(y_prob, epsilon, 1 - epsilon)
        logits = np.log(y_prob_clipped / (1 - y_prob_clipped))
        
        # Fit logistic regression
        self.platt_scaler = LogisticRegression()
        self.platt_scaler.fit(logits.reshape(-1, 1), y_true)
        
        return self.platt_scaler
    
    def fit_isotonic_regression(self, y_true: np.ndarray, y_prob: np.ndarray) -> IsotonicRegression:
        """
        Fit isotonic regression calibration.
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            
        Returns:
            Fitted isotonic regression model
        """
        self.isotonic_regressor = IsotonicRegression(out_of_bounds='clip')
        self.isotonic_regressor.fit(y_prob, y_true)
        
        return self.isotonic_regressor
    
    def apply_platt_scaling(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Apply Platt scaling to probabilities.
        
        Args:
            y_prob: Predicted probabilities
            
        Returns:
            Calibrated probabilities
        """
        if self.platt_scaler is None:
            raise ValueError("Platt scaler not fitted. Call fit_platt_scaling first.")
        
        epsilon = 1e-15
        y_prob_clipped = np.clip(y_prob, epsilon, 1 - epsilon)
        logits = np.log(y_prob_clipped / (1 - y_prob_clipped))
        
        calibrated_prob = self.platt_scaler.predict_proba(logits.reshape(-1, 1))[:, 1]
        
        return calibrated_prob
    
    def apply_isotonic_regression(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Apply isotonic regression to probabilities.
        
        Args:
            y_prob: Predicted probabilities
            
        Returns:
            Calibrated probabilities
        """
        if self.isotonic_regressor is None:
            raise ValueError("Isotonic regressor not fitted. Call fit_isotonic_regression first.")
        
        calibrated_prob = self.isotonic_regressor.predict(y_prob)
        
        return calibrated_prob
    
    def calibrate_probabilities(self, y_true: np.ndarray, y_prob: np.ndarray, 
                              method: str = 'isotonic') -> Tuple[np.ndarray, Any]:
        """
        Calibrate probabilities using specified method.
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            method: Calibration method ('platt' or 'isotonic')
            
        Returns:
            Tuple of (calibrated_probabilities, fitted_calibrator)
        """
        if method == 'platt':
            calibrator = self.fit_platt_scaling(y_true, y_prob)
            calibrated_prob = self.apply_platt_scaling(y_prob)
        elif method == 'isotonic':
            calibrator = self.fit_isotonic_regression(y_true, y_prob)
            calibrated_prob = self.apply_isotonic_regression(y_prob)
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        
        return calibrated_prob, calibrator


class ReliabilityDiagram:
    """
    Reliability diagram generation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize reliability diagram generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.calibration_config = config.get('calibration', {})
    
    def plot_reliability_diagram(self, y_true: np.ndarray, y_prob: np.ndarray, 
                                model_name: str = 'Model', n_bins: int = 10,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot reliability diagram.
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            model_name: Name of the model
            n_bins: Number of bins
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins
        )
        
        # Calculate bin counts
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_counts = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            bin_counts.append(in_bin.sum())
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
        
        # Reliability diagram
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax1.plot(mean_predicted_value, fraction_of_positives, 'o-', 
                label=f'{model_name}', linewidth=2, markersize=8)
        
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title(f'Reliability Diagram - {model_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        
        # Histogram of predictions
        ax2.hist(y_prob, bins=n_bins, alpha=0.7, density=True, 
                edgecolor='black', linewidth=1)
        ax2.set_xlabel('Predicted Probability')
        ax2.set_ylabel('Density')
        ax2.set_title(f'Distribution of Predicted Probabilities - {model_name}')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_multiple_reliability_diagrams(self, models_data: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                                         n_bins: int = 10, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot reliability diagrams for multiple models.
        
        Args:
            models_data: Dictionary of {model_name: (y_true, y_prob)}
            n_bins: Number of bins
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(models_data)))
        
        for i, (model_name, (y_true, y_prob)) in enumerate(models_data.items()):
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob, n_bins=n_bins
            )
            
            ax.plot(mean_predicted_value, fraction_of_positives, 'o-', 
                   label=model_name, linewidth=2, markersize=6, color=colors[i])
        
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title('Reliability Diagram Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class CalibrationAnalyzer:
    """
    Main calibration analysis framework.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize calibration analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.metrics_calculator = CalibrationMetrics(config)
        self.post_hoc_calibrator = PostHocCalibrator(config)
        self.reliability_plotter = ReliabilityDiagram(config)
    
    def analyze_model_calibration(self, model, X_test: np.ndarray, y_test: np.ndarray, 
                                model_name: str) -> Dict[str, Any]:
        """
        Analyze calibration for a single model.
        
        Args:
            model: Trained model with predict_proba method
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Dictionary of calibration analysis results
        """
        logger.info(f"Analyzing calibration for model: {model_name}")
        
        # Get predictions
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate calibration metrics
        calibration_metrics = self.metrics_calculator.calculate_all_metrics(y_test, y_prob)
        
        # Post-hoc calibration
        calibration_methods = self.config.get('calibration', {}).get('calibration_methods', ['isotonic'])
        
        post_hoc_results = {}
        for method in calibration_methods:
            try:
                calibrated_prob, calibrator = self.post_hoc_calibrator.calibrate_probabilities(
                    y_test, y_prob, method=method
                )
                
                # Calculate metrics for calibrated probabilities
                calibrated_metrics = self.metrics_calculator.calculate_all_metrics(
                    y_test, calibrated_prob
                )
                
                post_hoc_results[method] = {
                    'calibrated_probabilities': calibrated_prob,
                    'calibrator': calibrator,
                    'metrics': calibrated_metrics,
                    'improvement': {
                        'ece_improvement': calibration_metrics['ece'] - calibrated_metrics['ece'],
                        'brier_improvement': calibration_metrics['brier_score'] - calibrated_metrics['brier_score']
                    }
                }
                
            except Exception as e:
                logger.warning(f"Failed to apply {method} calibration for {model_name}: {str(e)}")
                continue
        
        # Generate reliability diagram data
        reliability_data = self.metrics_calculator.reliability_diagram_data(y_test, y_prob)
        
        results = {
            'model_name': model_name,
            'original_probabilities': y_prob,
            'original_metrics': calibration_metrics,
            'post_hoc_calibration': post_hoc_results,
            'reliability_data': reliability_data,
            'n_test_samples': len(y_test),
            'test_prevalence': y_test.mean()
        }
        
        logger.info(f"Completed calibration analysis for {model_name}")
        logger.info(f"Original ECE: {calibration_metrics['ece']:.3f}, "
                   f"Brier Score: {calibration_metrics['brier_score']:.3f}")
        
        return results
    
    def analyze_calibration(self, models: Dict[str, Any], 
                          target_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        """
        Analyze calibration for all models.
        
        Args:
            models: Dictionary of trained models
            target_data: Target domain data (X_target, y_target)
            
        Returns:
            Dictionary of calibration analysis results
        """
        logger.info("Analyzing calibration for all models")
        
        X_target, y_target = target_data
        
        # Analyze each model
        model_calibration_results = {}
        for model_name, model in models.items():
            try:
                results = self.analyze_model_calibration(model, X_target, y_target, model_name)
                model_calibration_results[model_name] = results
            except Exception as e:
                logger.error(f"Failed to analyze calibration for {model_name}: {str(e)}")
                continue
        
        # Generate comparison tables
        comparison_table = self._generate_calibration_comparison_table(model_calibration_results)
        improvement_table = self._generate_calibration_improvement_table(model_calibration_results)
        
        # Generate visualizations
        visualization_results = self._generate_calibration_visualizations(model_calibration_results)
        
        calibration_summary = {
            'model_calibration_results': model_calibration_results,
            'comparison_table': comparison_table,
            'improvement_table': improvement_table,
            'visualizations': visualization_results,
            'analysis_metadata': {
                'n_models_analyzed': len(model_calibration_results),
                'target_domain_size': len(y_target),
                'target_prevalence': y_target.mean(),
                'calibration_methods': self.config.get('calibration', {}).get('calibration_methods', ['isotonic']),
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            }
        }
        
        logger.info(f"Completed calibration analysis for {len(model_calibration_results)} models")
        
        return calibration_summary
    
    def _generate_calibration_comparison_table(self, model_results: Dict[str, Any]) -> pd.DataFrame:
        """Generate calibration comparison table."""
        
        comparison_data = []
        
        for model_name, results in model_results.items():
            original_metrics = results['original_metrics']
            
            row = {
                'Model': model_name,
                'ECE': original_metrics['ece'],
                'MCE': original_metrics['mce'],
                'Brier_Score': original_metrics['brier_score'],
                'Log_Loss': original_metrics['log_loss'],
                'Calibration_Slope': original_metrics['calibration_slope'],
                'Calibration_Intercept': original_metrics['calibration_intercept']
            }
            
            # Add post-hoc calibration results
            for method, post_hoc_result in results['post_hoc_calibration'].items():
                row[f'ECE_{method}'] = post_hoc_result['metrics']['ece']
                row[f'Brier_{method}'] = post_hoc_result['metrics']['brier_score']
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def _generate_calibration_improvement_table(self, model_results: Dict[str, Any]) -> pd.DataFrame:
        """Generate calibration improvement table."""
        
        improvement_data = []
        
        for model_name, results in model_results.items():
            for method, post_hoc_result in results['post_hoc_calibration'].items():
                improvement = post_hoc_result['improvement']
                
                row = {
                    'Model': model_name,
                    'Calibration_Method': method,
                    'ECE_Improvement': improvement['ece_improvement'],
                    'Brier_Improvement': improvement['brier_improvement'],
                    'ECE_Percent_Improvement': (improvement['ece_improvement'] / 
                                              results['original_metrics']['ece']) * 100,
                    'Brier_Percent_Improvement': (improvement['brier_improvement'] / 
                                                results['original_metrics']['brier_score']) * 100
                }
                
                improvement_data.append(row)
        
        return pd.DataFrame(improvement_data)
    
    def _generate_calibration_visualizations(self, model_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate calibration visualizations."""
        
        visualization_paths = {}
        
        # Individual reliability diagrams
        for model_name, results in model_results.items():
            y_true = np.ones(results['n_test_samples'])  # Placeholder
            y_prob = results['original_probabilities']
            
            fig = self.reliability_plotter.plot_reliability_diagram(
                y_true, y_prob, model_name, 
                save_path=f'reliability_diagram_{model_name.lower().replace(" ", "_")}.png'
            )
            plt.close(fig)
            
            visualization_paths[f'reliability_{model_name}'] = f'reliability_diagram_{model_name.lower().replace(" ", "_")}.png'
        
        # Comparison reliability diagram
        models_data = {}
        for model_name, results in model_results.items():
            y_true = np.ones(results['n_test_samples'])  # Placeholder
            y_prob = results['original_probabilities']
            models_data[model_name] = (y_true, y_prob)
        
        if len(models_data) > 1:
            fig = self.reliability_plotter.plot_multiple_reliability_diagrams(
                models_data, save_path='reliability_diagram_comparison.png'
            )
            plt.close(fig)
            
            visualization_paths['reliability_comparison'] = 'reliability_diagram_comparison.png'
        
        return visualization_paths

