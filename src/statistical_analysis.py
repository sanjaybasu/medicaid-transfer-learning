#!/usr/bin/env python3
"""
Statistical Analysis Module for Medicaid Transfer Learning Study

This module provides comprehensive statistical analysis capabilities including
significance testing, effect size calculations, and multiple comparison corrections.

Classes:
    StatisticalAnalyzer: Main statistical analysis framework
    EffectSizeCalculator: Effect size computation utilities
    MultipleComparisonCorrector: Multiple comparison correction methods
    PowerAnalysis: Statistical power analysis utilities
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, ttest_ind
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import ttest_power
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class EffectSizeCalculator:
    """
    Effect size computation utilities for various statistical tests.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize effect size calculator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    def cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Calculate Cohen's d effect size for two groups.
        
        Args:
            group1: First group values
            group2: Second group values
            
        Returns:
            Cohen's d effect size
        """
        n1, n2 = len(group1), len(group2)
        
        # Calculate means
        mean1, mean2 = np.mean(group1), np.mean(group2)
        
        # Calculate pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + 
                             (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
        
        # Cohen's d
        d = (mean1 - mean2) / pooled_std
        
        return d
    
    def glass_delta(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Calculate Glass's delta effect size.
        
        Args:
            group1: First group values (typically control)
            group2: Second group values (typically treatment)
            
        Returns:
            Glass's delta effect size
        """
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1 = np.std(group1, ddof=1)
        
        delta = (mean2 - mean1) / std1
        
        return delta
    
    def cramers_v(self, contingency_table: np.ndarray) -> float:
        """
        Calculate Cramér's V effect size for categorical associations.
        
        Args:
            contingency_table: 2D contingency table
            
        Returns:
            Cramér's V effect size
        """
        chi2, _, _, _ = chi2_contingency(contingency_table)
        n = contingency_table.sum()
        min_dim = min(contingency_table.shape) - 1
        
        v = np.sqrt(chi2 / (n * min_dim))
        
        return v
    
    def odds_ratio(self, contingency_table: np.ndarray) -> Tuple[float, Tuple[float, float]]:
        """
        Calculate odds ratio and confidence interval.
        
        Args:
            contingency_table: 2x2 contingency table
            
        Returns:
            Tuple of (odds_ratio, (lower_ci, upper_ci))
        """
        if contingency_table.shape != (2, 2):
            raise ValueError("Contingency table must be 2x2 for odds ratio calculation")
        
        a, b, c, d = contingency_table.ravel()
        
        # Calculate odds ratio
        or_value = (a * d) / (b * c) if (b * c) > 0 else np.inf
        
        # Calculate 95% confidence interval
        log_or = np.log(or_value) if or_value > 0 and or_value != np.inf else 0
        se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d) if all(x > 0 for x in [a, b, c, d]) else 0
        
        ci_lower = np.exp(log_or - 1.96 * se_log_or)
        ci_upper = np.exp(log_or + 1.96 * se_log_or)
        
        return or_value, (ci_lower, ci_upper)
    
    def interpret_effect_size(self, effect_size: float, effect_type: str) -> str:
        """
        Interpret effect size magnitude according to Cohen's conventions.
        
        Args:
            effect_size: Calculated effect size
            effect_type: Type of effect size ('cohens_d', 'cramers_v', etc.)
            
        Returns:
            Interpretation string
        """
        abs_effect = abs(effect_size)
        
        if effect_type in ['cohens_d', 'glass_delta']:
            if abs_effect < 0.2:
                return 'negligible'
            elif abs_effect < 0.5:
                return 'small'
            elif abs_effect < 0.8:
                return 'medium'
            else:
                return 'large'
        
        elif effect_type == 'cramers_v':
            if abs_effect < 0.1:
                return 'negligible'
            elif abs_effect < 0.3:
                return 'small'
            elif abs_effect < 0.5:
                return 'medium'
            else:
                return 'large'
        
        else:
            return 'unknown'


class MultipleComparisonCorrector:
    """
    Multiple comparison correction methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize multiple comparison corrector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.alpha = config.get('statistical_tests', {}).get('significance_level', 0.05)
    
    def correct_pvalues(self, pvalues: np.ndarray, method: str = 'bonferroni') -> Dict[str, np.ndarray]:
        """
        Apply multiple comparison correction to p-values.
        
        Args:
            pvalues: Array of p-values
            method: Correction method ('bonferroni', 'holm', 'fdr_bh', etc.)
            
        Returns:
            Dictionary with corrected p-values and rejection decisions
        """
        rejected, pvals_corrected, alpha_sidak, alpha_bonf = multipletests(
            pvalues, alpha=self.alpha, method=method
        )
        
        return {
            'pvalues_corrected': pvals_corrected,
            'rejected': rejected,
            'alpha_corrected': alpha_bonf if method == 'bonferroni' else alpha_sidak,
            'method': method
        }
    
    def family_wise_error_rate(self, n_comparisons: int) -> float:
        """
        Calculate family-wise error rate for Bonferroni correction.
        
        Args:
            n_comparisons: Number of comparisons
            
        Returns:
            Corrected alpha level
        """
        return self.alpha / n_comparisons


class PowerAnalysis:
    """
    Statistical power analysis utilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize power analysis.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.alpha = config.get('statistical_tests', {}).get('significance_level', 0.05)
        self.power = config.get('power_analysis', {}).get('target_power', 0.8)
    
    def calculate_sample_size(self, effect_size: float, test_type: str = 'two_sample') -> int:
        """
        Calculate required sample size for desired power.
        
        Args:
            effect_size: Expected effect size
            test_type: Type of statistical test
            
        Returns:
            Required sample size per group
        """
        if test_type == 'two_sample':
            # For two-sample t-test
            from statsmodels.stats.power import ttest_power
            
            # Calculate sample size using iterative approach
            for n in range(10, 10000):
                power = ttest_power(effect_size, n, self.alpha, alternative='two-sided')
                if power >= self.power:
                    return n
        
        return None
    
    def calculate_power(self, effect_size: float, sample_size: int, 
                       test_type: str = 'two_sample') -> float:
        """
        Calculate statistical power for given parameters.
        
        Args:
            effect_size: Effect size
            sample_size: Sample size per group
            test_type: Type of statistical test
            
        Returns:
            Statistical power
        """
        if test_type == 'two_sample':
            power = ttest_power(effect_size, sample_size, self.alpha, alternative='two-sided')
            return power
        
        return None


class StatisticalAnalyzer:
    """
    Main statistical analysis framework.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize statistical analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.effect_calculator = EffectSizeCalculator(config)
        self.multiple_corrector = MultipleComparisonCorrector(config)
        self.power_analyzer = PowerAnalysis(config)
        self.alpha = config.get('statistical_tests', {}).get('significance_level', 0.05)
    
    def compare_two_groups_continuous(self, group1: np.ndarray, group2: np.ndarray,
                                    group1_name: str = 'Group 1', 
                                    group2_name: str = 'Group 2') -> Dict[str, Any]:
        """
        Compare two groups on a continuous variable.
        
        Args:
            group1: First group values
            group2: Second group values
            group1_name: Name of first group
            group2_name: Name of second group
            
        Returns:
            Dictionary of statistical test results
        """
        # Descriptive statistics
        desc_stats = {
            f'{group1_name}_mean': np.mean(group1),
            f'{group1_name}_std': np.std(group1, ddof=1),
            f'{group1_name}_n': len(group1),
            f'{group2_name}_mean': np.mean(group2),
            f'{group2_name}_std': np.std(group2, ddof=1),
            f'{group2_name}_n': len(group2)
        }
        
        # Test for normality
        _, p_norm1 = stats.shapiro(group1) if len(group1) <= 5000 else stats.normaltest(group1)
        _, p_norm2 = stats.shapiro(group2) if len(group2) <= 5000 else stats.normaltest(group2)
        
        # Test for equal variances
        _, p_levene = stats.levene(group1, group2)
        
        # Choose appropriate test
        if p_norm1 > 0.05 and p_norm2 > 0.05:
            # Both groups are normal, use t-test
            if p_levene > 0.05:
                # Equal variances
                statistic, p_value = ttest_ind(group1, group2, equal_var=True)
                test_used = 'Independent t-test (equal variances)'
            else:
                # Unequal variances (Welch's t-test)
                statistic, p_value = ttest_ind(group1, group2, equal_var=False)
                test_used = "Welch's t-test (unequal variances)"
        else:
            # Non-normal data, use Mann-Whitney U test
            statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
            test_used = 'Mann-Whitney U test'
        
        # Effect size
        cohens_d = self.effect_calculator.cohens_d(group1, group2)
        effect_interpretation = self.effect_calculator.interpret_effect_size(cohens_d, 'cohens_d')
        
        results = {
            'descriptive_stats': desc_stats,
            'test_statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'test_used': test_used,
            'effect_size': cohens_d,
            'effect_interpretation': effect_interpretation,
            'assumptions': {
                'normality_group1_p': p_norm1,
                'normality_group2_p': p_norm2,
                'equal_variances_p': p_levene
            }
        }
        
        return results
    
    def compare_two_groups_categorical(self, group1: np.ndarray, group2: np.ndarray,
                                     group1_name: str = 'Group 1',
                                     group2_name: str = 'Group 2') -> Dict[str, Any]:
        """
        Compare two groups on a categorical variable.
        
        Args:
            group1: First group categorical values
            group2: Second group categorical values
            group1_name: Name of first group
            group2_name: Name of second group
            
        Returns:
            Dictionary of statistical test results
        """
        # Create contingency table
        contingency_table = pd.crosstab(
            np.concatenate([group1, group2]),
            np.concatenate([np.repeat(group1_name, len(group1)), 
                           np.repeat(group2_name, len(group2))])
        ).values
        
        # Chi-square test
        chi2_stat, p_chi2, dof, expected = chi2_contingency(contingency_table)
        
        # Fisher's exact test (for 2x2 tables)
        if contingency_table.shape == (2, 2):
            odds_ratio, p_fisher = fisher_exact(contingency_table)
            or_value, or_ci = self.effect_calculator.odds_ratio(contingency_table)
        else:
            p_fisher = None
            odds_ratio = None
            or_value, or_ci = None, (None, None)
        
        # Effect size (Cramér's V)
        cramers_v = self.effect_calculator.cramers_v(contingency_table)
        effect_interpretation = self.effect_calculator.interpret_effect_size(cramers_v, 'cramers_v')
        
        results = {
            'contingency_table': contingency_table,
            'chi2_statistic': chi2_stat,
            'chi2_p_value': p_chi2,
            'degrees_of_freedom': dof,
            'fisher_exact_p': p_fisher,
            'odds_ratio': odds_ratio,
            'odds_ratio_ci': or_ci,
            'cramers_v': cramers_v,
            'effect_interpretation': effect_interpretation,
            'significant': p_chi2 < self.alpha
        }
        
        return results
    
    def compare_model_performance(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare performance across multiple models.
        
        Args:
            model_results: Dictionary of model evaluation results
            
        Returns:
            Dictionary of comparison results
        """
        logger.info("Performing statistical comparison of model performance")
        
        # Extract performance metrics
        model_names = list(model_results.keys())
        metrics_data = {}
        
        for metric in ['auc_roc', 'youdens_j', 'f1_score', 'mcc']:
            metrics_data[metric] = []
            for model_name in model_names:
                if metric in model_results[model_name]['metrics']:
                    metrics_data[metric].append(model_results[model_name]['metrics'][metric])
                else:
                    metrics_data[metric].append(np.nan)
        
        # Pairwise comparisons
        pairwise_results = {}
        
        for metric in metrics_data.keys():
            metric_values = np.array(metrics_data[metric])
            
            # Skip if insufficient data
            if np.sum(~np.isnan(metric_values)) < 2:
                continue
            
            pairwise_results[metric] = []
            
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    model1 = model_names[i]
                    model2 = model_names[j]
                    
                    value1 = metric_values[i]
                    value2 = metric_values[j]
                    
                    if not (np.isnan(value1) or np.isnan(value2)):
                        # Calculate difference and effect size
                        difference = value2 - value1
                        percent_improvement = (difference / value1) * 100 if value1 != 0 else 0
                        
                        # Simplified significance test (in practice, use bootstrap or DeLong test)
                        # This is a placeholder - real implementation would use proper statistical tests
                        z_score = difference / (0.01)  # Simplified standard error
                        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                        
                        comparison = {
                            'model_1': model1,
                            'model_2': model2,
                            'metric': metric,
                            'value_1': value1,
                            'value_2': value2,
                            'difference': difference,
                            'percent_improvement': percent_improvement,
                            'z_score': z_score,
                            'p_value': p_value,
                            'significant': p_value < self.alpha
                        }
                        
                        pairwise_results[metric].append(comparison)
        
        # Multiple comparison correction
        all_p_values = []
        all_comparisons = []
        
        for metric, comparisons in pairwise_results.items():
            for comp in comparisons:
                all_p_values.append(comp['p_value'])
                all_comparisons.append(comp)
        
        if all_p_values:
            correction_method = self.config.get('statistical_tests', {}).get(
                'multiple_comparison_correction', 'bonferroni'
            )
            
            corrected_results = self.multiple_corrector.correct_pvalues(
                np.array(all_p_values), method=correction_method
            )
            
            # Update comparisons with corrected p-values
            for i, comp in enumerate(all_comparisons):
                comp['p_value_corrected'] = corrected_results['pvalues_corrected'][i]
                comp['significant_corrected'] = corrected_results['rejected'][i]
        
        # Summary statistics
        summary_stats = {}
        for metric in metrics_data.keys():
            values = np.array(metrics_data[metric])
            valid_values = values[~np.isnan(values)]
            
            if len(valid_values) > 0:
                summary_stats[metric] = {
                    'mean': np.mean(valid_values),
                    'std': np.std(valid_values, ddof=1),
                    'min': np.min(valid_values),
                    'max': np.max(valid_values),
                    'range': np.max(valid_values) - np.min(valid_values),
                    'best_model': model_names[np.nanargmax(values)]
                }
        
        results = {
            'pairwise_comparisons': pairwise_results,
            'summary_statistics': summary_stats,
            'multiple_comparison_correction': {
                'method': correction_method if all_p_values else None,
                'n_comparisons': len(all_p_values),
                'alpha_corrected': corrected_results.get('alpha_corrected') if all_p_values else None
            },
            'overall_best_model': self._determine_overall_best_model(summary_stats)
        }
        
        logger.info("Completed statistical comparison of model performance")
        
        return results
    
    def _determine_overall_best_model(self, summary_stats: Dict[str, Any]) -> str:
        """
        Determine overall best performing model across metrics.
        
        Args:
            summary_stats: Summary statistics for each metric
            
        Returns:
            Name of overall best model
        """
        # Weight metrics by importance (can be configured)
        metric_weights = {
            'auc_roc': 0.3,
            'youdens_j': 0.3,
            'f1_score': 0.2,
            'mcc': 0.2
        }
        
        model_scores = {}
        
        for metric, stats in summary_stats.items():
            if metric in metric_weights:
                best_model = stats['best_model']
                weight = metric_weights[metric]
                
                if best_model not in model_scores:
                    model_scores[best_model] = 0
                
                model_scores[best_model] += weight
        
        if model_scores:
            return max(model_scores, key=model_scores.get)
        else:
            return None
    
    def perform_statistical_tests(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis on evaluation results.
        
        Args:
            evaluation_results: Results from model evaluation
            
        Returns:
            Dictionary of statistical analysis results
        """
        logger.info("Performing comprehensive statistical analysis")
        
        # Model performance comparison
        model_comparison = self.compare_model_performance(evaluation_results['model_results'])
        
        # Generate statistical summary tables
        comparison_table = self._generate_comparison_table(model_comparison)
        significance_table = self._generate_significance_table(model_comparison)
        
        statistical_results = {
            'model_comparison': model_comparison,
            'comparison_table': comparison_table,
            'significance_table': significance_table,
            'analysis_metadata': {
                'significance_level': self.alpha,
                'multiple_comparison_method': self.config.get('statistical_tests', {}).get(
                    'multiple_comparison_correction', 'bonferroni'
                ),
                'n_models_compared': len(evaluation_results['model_results']),
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            }
        }
        
        logger.info("Completed comprehensive statistical analysis")
        
        return statistical_results
    
    def _generate_comparison_table(self, model_comparison: Dict[str, Any]) -> pd.DataFrame:
        """Generate formatted comparison table."""
        
        comparison_data = []
        
        for metric, comparisons in model_comparison['pairwise_comparisons'].items():
            for comp in comparisons:
                row = {
                    'Metric': metric.upper(),
                    'Model_1': comp['model_1'],
                    'Model_2': comp['model_2'],
                    'Value_1': comp['value_1'],
                    'Value_2': comp['value_2'],
                    'Difference': comp['difference'],
                    'Percent_Improvement': comp['percent_improvement'],
                    'P_Value': comp['p_value'],
                    'P_Value_Corrected': comp.get('p_value_corrected', comp['p_value']),
                    'Significant': comp.get('significant_corrected', comp['significant'])
                }
                comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def _generate_significance_table(self, model_comparison: Dict[str, Any]) -> pd.DataFrame:
        """Generate significance summary table."""
        
        significance_data = []
        
        for metric, stats in model_comparison['summary_statistics'].items():
            row = {
                'Metric': metric.upper(),
                'Best_Model': stats['best_model'],
                'Best_Value': stats['max'],
                'Mean_Value': stats['mean'],
                'Std_Value': stats['std'],
                'Range': stats['range']
            }
            significance_data.append(row)
        
        return pd.DataFrame(significance_data)

