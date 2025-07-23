#!/usr/bin/env python3
"""
Visualization Module for Medicaid Transfer Learning Study

This module provides comprehensive visualization capabilities for transfer learning
analysis results including performance comparisons, calibration plots, and ablation studies.

Classes:
    VisualizationGenerator: Main visualization framework
    PerformancePlotter: Performance comparison visualizations
    CalibrationPlotter: Calibration analysis visualizations
    AblationPlotter: Ablation study visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Set publication-quality style
plt.style.use('default')
sns.set_palette("husl")

# Publication parameters
FIGURE_DPI = 300
FIGURE_SIZE_SINGLE = (8, 6)
FIGURE_SIZE_DOUBLE = (12, 8)
FONT_SIZE_TITLE = 14
FONT_SIZE_LABEL = 12
FONT_SIZE_TICK = 10
FONT_SIZE_LEGEND = 10


class PerformancePlotter:
    """
    Performance comparison visualizations.
    """
    
    def __init__(self, config: Dict[str, Any], output_dir: Path):
        """
        Initialize performance plotter.
        
        Args:
            config: Configuration dictionary
            output_dir: Output directory for figures
        """
        self.config = config
        self.output_dir = output_dir
        self.figures_dir = output_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame, 
                            save_name: str = 'model_comparison') -> str:
        """
        Plot comprehensive model comparison.
        
        Args:
            comparison_df: DataFrame with model comparison results
            save_name: Name for saved figure
            
        Returns:
            Path to saved figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        models = comparison_df['Model']
        
        # AUC comparison
        ax1.bar(range(len(models)), comparison_df['AUC'], 
               color='skyblue', alpha=0.8, edgecolor='black')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('AUC-ROC')
        ax1.set_title('AUC-ROC Comparison')
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(comparison_df['AUC']):
            ax1.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Youden's J comparison
        ax2.bar(range(len(models)), comparison_df['Youdens_J'], 
               color='lightcoral', alpha=0.8, edgecolor='black')
        ax2.set_xlabel('Models')
        ax2.set_ylabel("Youden's J Index")
        ax2.set_title("Youden's J Index Comparison")
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(comparison_df['Youdens_J']):
            ax2.text(i, v + 0.002, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # F1-Score comparison
        ax3.bar(range(len(models)), comparison_df['F1_Score'], 
               color='lightgreen', alpha=0.8, edgecolor='black')
        ax3.set_xlabel('Models')
        ax3.set_ylabel('F1-Score')
        ax3.set_title('F1-Score Comparison')
        ax3.set_xticks(range(len(models)))
        ax3.set_xticklabels(models, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(comparison_df['F1_Score']):
            ax3.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # NNT comparison (lower is better)
        ax4.bar(range(len(models)), comparison_df['NNT'], 
               color='gold', alpha=0.8, edgecolor='black')
        ax4.set_xlabel('Models')
        ax4.set_ylabel('Number Needed to Treat')
        ax4.set_title('Number Needed to Treat (Lower is Better)')
        ax4.set_xticks(range(len(models)))
        ax4.set_xticklabels(models, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(comparison_df['NNT']):
            ax4.text(i, v + 1, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.figures_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved model comparison plot: {save_path}")
        return str(save_path)
    
    def plot_improvement_analysis(self, improvement_df: pd.DataFrame,
                                save_name: str = 'improvement_analysis') -> str:
        """
        Plot improvement analysis over baseline.
        
        Args:
            improvement_df: DataFrame with improvement metrics
            save_name: Name for saved figure
            
        Returns:
            Path to saved figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZE_DOUBLE)
        
        models = improvement_df['Model']
        
        # Youden's J improvement
        improvements = improvement_df['Youdens_J_Improvement']
        colors = ['red' if x < 0 else 'green' for x in improvements]
        
        bars1 = ax1.barh(range(len(models)), improvements, color=colors, alpha=0.7)
        ax1.set_xlabel('Improvement in Youden\'s J (%)')
        ax1.set_ylabel('Models')
        ax1.set_title('Performance Improvement Over Baseline')
        ax1.set_yticks(range(len(models)))
        ax1.set_yticklabels(models)
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Add value labels
        for i, v in enumerate(improvements):
            ax1.text(v + (5 if v >= 0 else -5), i, f'{v:.1f}%', 
                    ha='left' if v >= 0 else 'right', va='center', fontweight='bold')
        
        # Multiple metrics improvement
        metrics = ['AUC_Improvement', 'F1_Improvement', 'MCC_Improvement']
        metric_labels = ['AUC', 'F1-Score', 'MCC']
        
        x = np.arange(len(models))
        width = 0.25
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            if metric in improvement_df.columns:
                values = improvement_df[metric]
                ax2.bar(x + i*width, values, width, label=label, alpha=0.8)
        
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Improvement (%)')
        ax2.set_title('Multi-Metric Improvement Analysis')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.figures_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved improvement analysis plot: {save_path}")
        return str(save_path)
    
    def plot_sensitivity_specificity(self, comparison_df: pd.DataFrame,
                                   save_name: str = 'sensitivity_specificity') -> str:
        """
        Plot sensitivity vs specificity scatter plot.
        
        Args:
            comparison_df: DataFrame with model comparison results
            save_name: Name for saved figure
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=FIGURE_SIZE_SINGLE)
        
        # Create scatter plot
        scatter = ax.scatter(comparison_df['Specificity'], comparison_df['Sensitivity'], 
                           s=200, alpha=0.7, c=range(len(comparison_df)), 
                           cmap='viridis', edgecolors='black', linewidth=1)
        
        # Add model labels
        for i, model in enumerate(comparison_df['Model']):
            ax.annotate(model, 
                       (comparison_df.iloc[i]['Specificity'], comparison_df.iloc[i]['Sensitivity']),
                       xytext=(5, 5), textcoords='offset points', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Add diagonal line for reference
        ax.plot([0, 1], [1, 0], 'k--', alpha=0.5, label='Random Performance')
        
        ax.set_xlabel('Specificity')
        ax.set_ylabel('Sensitivity')
        ax.set_title('Sensitivity vs Specificity Trade-off')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.figures_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved sensitivity-specificity plot: {save_path}")
        return str(save_path)


class CalibrationPlotter:
    """
    Calibration analysis visualizations.
    """
    
    def __init__(self, config: Dict[str, Any], output_dir: Path):
        """
        Initialize calibration plotter.
        
        Args:
            config: Configuration dictionary
            output_dir: Output directory for figures
        """
        self.config = config
        self.output_dir = output_dir
        self.figures_dir = output_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
    
    def plot_calibration_comparison(self, calibration_df: pd.DataFrame,
                                  save_name: str = 'calibration_comparison') -> str:
        """
        Plot calibration metrics comparison.
        
        Args:
            calibration_df: DataFrame with calibration results
            save_name: Name for saved figure
            
        Returns:
            Path to saved figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        models = calibration_df['Model']
        
        # ECE comparison
        ax1.bar(range(len(models)), calibration_df['ECE'], 
               color='lightblue', alpha=0.8, edgecolor='black')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Expected Calibration Error')
        ax1.set_title('Expected Calibration Error (Lower is Better)')
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(calibration_df['ECE']):
            ax1.text(i, v + 0.002, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Brier Score comparison
        ax2.bar(range(len(models)), calibration_df['Brier_Score'], 
               color='lightcoral', alpha=0.8, edgecolor='black')
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Brier Score')
        ax2.set_title('Brier Score (Lower is Better)')
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(calibration_df['Brier_Score']):
            ax2.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Calibration slope
        ax3.bar(range(len(models)), calibration_df['Calibration_Slope'], 
               color='lightgreen', alpha=0.8, edgecolor='black')
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Calibration Slope')
        ax3.set_title('Calibration Slope (1.0 is Perfect)')
        ax3.set_xticks(range(len(models)))
        ax3.set_xticklabels(models, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Perfect Calibration')
        ax3.legend()
        
        # Add value labels
        for i, v in enumerate(calibration_df['Calibration_Slope']):
            ax3.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Post-hoc calibration improvement (if available)
        if 'ECE_isotonic' in calibration_df.columns:
            original_ece = calibration_df['ECE']
            improved_ece = calibration_df['ECE_isotonic']
            improvement = ((original_ece - improved_ece) / original_ece) * 100
            
            ax4.bar(range(len(models)), improvement, 
                   color='gold', alpha=0.8, edgecolor='black')
            ax4.set_xlabel('Models')
            ax4.set_ylabel('ECE Improvement (%)')
            ax4.set_title('Post-hoc Calibration Improvement')
            ax4.set_xticks(range(len(models)))
            ax4.set_xticklabels(models, rotation=45, ha='right')
            ax4.grid(True, alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(improvement):
                ax4.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.figures_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved calibration comparison plot: {save_path}")
        return str(save_path)


class AblationPlotter:
    """
    Ablation study visualizations.
    """
    
    def __init__(self, config: Dict[str, Any], output_dir: Path):
        """
        Initialize ablation plotter.
        
        Args:
            config: Configuration dictionary
            output_dir: Output directory for figures
        """
        self.config = config
        self.output_dir = output_dir
        self.figures_dir = output_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
    
    def plot_feature_importance(self, feature_importance_df: pd.DataFrame,
                              save_name: str = 'feature_importance') -> str:
        """
        Plot feature importance analysis.
        
        Args:
            feature_importance_df: DataFrame with feature importance results
            save_name: Name for saved figure
            
        Returns:
            Path to saved figure
        """
        # Get top features across all models
        top_features = feature_importance_df.groupby('Feature_Index')['Importance_Mean'].mean().nlargest(20)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZE_DOUBLE)
        
        # Feature importance by model
        models = feature_importance_df['Model'].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
        
        for i, model in enumerate(models):
            model_data = feature_importance_df[feature_importance_df['Model'] == model]
            top_model_features = model_data.nlargest(10, 'Importance_Mean')
            
            ax1.barh(range(len(top_model_features)), top_model_features['Importance_Mean'],
                    alpha=0.7, label=model, color=colors[i])
        
        ax1.set_xlabel('Feature Importance')
        ax1.set_ylabel('Feature Rank')
        ax1.set_title('Top Feature Importance by Model')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Overall feature importance (averaged across models)
        overall_importance = feature_importance_df.groupby('Feature_Index').agg({
            'Importance_Mean': 'mean',
            'Importance_Std': 'mean'
        }).nlargest(15, 'Importance_Mean')
        
        ax2.barh(range(len(overall_importance)), overall_importance['Importance_Mean'],
                xerr=overall_importance['Importance_Std'], alpha=0.8, 
                color='skyblue', edgecolor='black')
        ax2.set_xlabel('Average Feature Importance')
        ax2.set_ylabel('Feature Rank')
        ax2.set_title('Overall Feature Importance (Averaged Across Models)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.figures_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved feature importance plot: {save_path}")
        return str(save_path)
    
    def plot_grouped_importance(self, grouped_importance_df: pd.DataFrame,
                              save_name: str = 'grouped_importance') -> str:
        """
        Plot grouped feature importance analysis.
        
        Args:
            grouped_importance_df: DataFrame with grouped importance results
            save_name: Name for saved figure
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=FIGURE_SIZE_SINGLE)
        
        # Pivot data for plotting
        pivot_data = grouped_importance_df.pivot(index='Feature_Group', 
                                               columns='Model', 
                                               values='Importance_Mean')
        
        # Create grouped bar plot
        pivot_data.plot(kind='bar', ax=ax, alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Feature Groups')
        ax.set_ylabel('Group Importance')
        ax.set_title('Feature Group Importance Across Models')
        ax.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save figure
        save_path = self.figures_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved grouped importance plot: {save_path}")
        return str(save_path)
    
    def plot_component_ablation(self, component_ablation_df: pd.DataFrame,
                              save_name: str = 'component_ablation') -> str:
        """
        Plot component ablation analysis.
        
        Args:
            component_ablation_df: DataFrame with component ablation results
            save_name: Name for saved figure
            
        Returns:
            Path to saved figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZE_DOUBLE)
        
        # AUC drop analysis
        models = component_ablation_df['Model'].unique()
        
        for model in models:
            model_data = component_ablation_df[component_ablation_df['Model'] == model]
            
            ax1.bar(range(len(model_data)), model_data['AUC_Drop_Percent'],
                   alpha=0.7, label=model, edgecolor='black')
        
        ax1.set_xlabel('Component Removed')
        ax1.set_ylabel('AUC Drop (%)')
        ax1.set_title('Performance Drop by Component Removal')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Multi-metric drop analysis
        metrics = ['AUC_Drop_Percent', 'F1_Drop_Percent', 'MCC_Drop_Percent']
        metric_labels = ['AUC', 'F1-Score', 'MCC']
        
        # Average across models for each component
        avg_drops = component_ablation_df.groupby('Component_Removed')[metrics].mean()
        
        x = np.arange(len(avg_drops))
        width = 0.25
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax2.bar(x + i*width, avg_drops[metric], width, label=label, alpha=0.8)
        
        ax2.set_xlabel('Component Removed')
        ax2.set_ylabel('Average Performance Drop (%)')
        ax2.set_title('Multi-Metric Performance Drop Analysis')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(avg_drops.index, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.figures_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved component ablation plot: {save_path}")
        return str(save_path)


class VisualizationGenerator:
    """
    Main visualization framework that orchestrates all visualization generation.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize visualization generator.
        
        Args:
            output_dir: Output directory for all visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize plotters
        self.performance_plotter = PerformancePlotter({}, self.output_dir)
        self.calibration_plotter = CalibrationPlotter({}, self.output_dir)
        self.ablation_plotter = AblationPlotter({}, self.output_dir)
    
    def generate_all_figures(self, performance_results: Dict[str, Any],
                           statistical_results: Dict[str, Any],
                           calibration_results: Dict[str, Any],
                           ablation_results: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Generate all visualization figures.
        
        Args:
            performance_results: Performance evaluation results
            statistical_results: Statistical analysis results
            calibration_results: Calibration analysis results
            ablation_results: Ablation study results
            
        Returns:
            Dictionary of generated figure paths by category
        """
        logger.info("Generating all visualization figures")
        
        generated_figures = {
            'performance': [],
            'calibration': [],
            'ablation': [],
            'statistical': []
        }
        
        try:
            # Performance visualizations
            if 'comparison_table' in performance_results:
                comparison_df = performance_results['comparison_table']
                
                # Model comparison
                fig_path = self.performance_plotter.plot_model_comparison(comparison_df)
                generated_figures['performance'].append(fig_path)
                
                # Sensitivity-specificity plot
                fig_path = self.performance_plotter.plot_sensitivity_specificity(comparison_df)
                generated_figures['performance'].append(fig_path)
            
            if 'improvement_metrics' in performance_results:
                improvement_df = performance_results['improvement_metrics']
                
                # Improvement analysis
                fig_path = self.performance_plotter.plot_improvement_analysis(improvement_df)
                generated_figures['performance'].append(fig_path)
            
            # Calibration visualizations
            if 'comparison_table' in calibration_results:
                calibration_df = calibration_results['comparison_table']
                
                # Calibration comparison
                fig_path = self.calibration_plotter.plot_calibration_comparison(calibration_df)
                generated_figures['calibration'].append(fig_path)
            
            # Ablation visualizations
            if 'summary_tables' in ablation_results:
                summary_tables = ablation_results['summary_tables']
                
                # Feature importance
                if 'feature_importance' in summary_tables:
                    fig_path = self.ablation_plotter.plot_feature_importance(
                        summary_tables['feature_importance']
                    )
                    generated_figures['ablation'].append(fig_path)
                
                # Grouped importance
                if 'grouped_importance' in summary_tables:
                    fig_path = self.ablation_plotter.plot_grouped_importance(
                        summary_tables['grouped_importance']
                    )
                    generated_figures['ablation'].append(fig_path)
                
                # Component ablation
                if 'component_ablation' in summary_tables:
                    fig_path = self.ablation_plotter.plot_component_ablation(
                        summary_tables['component_ablation']
                    )
                    generated_figures['ablation'].append(fig_path)
            
            # Statistical visualizations
            if 'comparison_table' in statistical_results:
                # Additional statistical plots can be added here
                pass
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
        
        # Generate summary report
        self._generate_visualization_summary(generated_figures)
        
        logger.info(f"Generated {sum(len(figs) for figs in generated_figures.values())} figures")
        
        return generated_figures
    
    def _generate_visualization_summary(self, generated_figures: Dict[str, List[str]]) -> None:
        """Generate a summary report of all generated visualizations."""
        
        summary_path = self.output_dir / "visualization_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("VISUALIZATION SUMMARY REPORT\n")
            f.write("=" * 40 + "\n\n")
            
            total_figures = sum(len(figs) for figs in generated_figures.values())
            f.write(f"Total figures generated: {total_figures}\n\n")
            
            for category, figure_paths in generated_figures.items():
                f.write(f"{category.upper()} FIGURES ({len(figure_paths)}):\n")
                f.write("-" * 30 + "\n")
                
                for i, fig_path in enumerate(figure_paths, 1):
                    fig_name = Path(fig_path).name
                    f.write(f"{i}. {fig_name}\n")
                
                f.write("\n")
            
            f.write("All figures are saved in both PNG (high resolution) and PDF formats.\n")
            f.write("Figures are located in the 'figures/' subdirectory.\n")
        
        logger.info(f"Visualization summary saved to {summary_path}")

