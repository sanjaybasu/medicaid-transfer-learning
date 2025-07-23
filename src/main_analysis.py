#!/usr/bin/env python3
"""
Main Analysis Script for Medicaid Transfer Learning Study

This script orchestrates the complete analysis pipeline including data preprocessing,
model training, evaluation, statistical analysis, calibration assessment, and ablation studies.

Usage:
    python main_analysis.py --config config/model_config.yaml --data-config config/data_config.yaml
"""

import argparse
import logging
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_preprocessing import DataPreprocessor
from transfer_learning_models import (
    SourceOnlyTransfer, PrototypicalNetworks, MAML, 
    DomainAdversarialNetwork, CausalTransferLearning, 
    TabTransformer, MetaEnsemble
)
from evaluation_metrics import EvaluationFramework
from statistical_analysis import StatisticalAnalyzer
from calibration_analysis import CalibrationAnalyzer
from ablation_study import AblationStudy
from visualization import VisualizationGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MedicaidTransferLearningAnalysis:
    """
    Main analysis class that orchestrates the complete transfer learning study.
    """
    
    def __init__(self, model_config_path: str, data_config_path: str, output_dir: str = "results"):
        """
        Initialize the analysis framework.
        
        Args:
            model_config_path: Path to model configuration file
            data_config_path: Path to data configuration file
            output_dir: Output directory for results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configurations
        self.model_config = self._load_config(model_config_path)
        self.data_config = self._load_config(data_config_path)
        
        # Initialize components
        self.data_preprocessor = DataPreprocessor(self.data_config)
        self.evaluation_framework = EvaluationFramework(self.model_config)
        self.statistical_analyzer = StatisticalAnalyzer(self.model_config)
        self.calibration_analyzer = CalibrationAnalyzer(self.model_config)
        self.ablation_study = AblationStudy(self.model_config)
        self.visualization_generator = VisualizationGenerator(self.output_dir)
        
        # Model registry
        self.model_registry = {
            'source_only': SourceOnlyTransfer,
            'prototypical_networks': PrototypicalNetworks,
            'maml': MAML,
            'domain_adversarial': DomainAdversarialNetwork,
            'causal_transfer': CausalTransferLearning,
            'tab_transformer': TabTransformer,
            'meta_ensemble': MetaEnsemble
        }
        
        logger.info("Initialized Medicaid Transfer Learning Analysis")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _set_random_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        seeds = self.model_config.get('random_seeds', {})
        
        np.random.seed(seeds.get('model_init', 42))
        
        # Set seeds for other libraries if available
        try:
            import torch
            torch.manual_seed(seeds.get('model_init', 42))
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seeds.get('model_init', 42))
        except ImportError:
            pass
        
        try:
            import tensorflow as tf
            tf.random.set_seed(seeds.get('model_init', 42))
        except ImportError:
            pass
        
        logger.info("Set random seeds for reproducibility")
    
    def load_and_preprocess_data(self) -> Tuple[Tuple[np.ndarray, np.ndarray], 
                                              Tuple[np.ndarray, np.ndarray]]:
        """
        Load and preprocess the data.
        
        Returns:
            Tuple of (source_data, target_data) where each is (X, y)
        """
        logger.info("Loading and preprocessing data")
        
        # Generate synthetic data for demonstration
        # In practice, this would load real Medicaid data
        source_data, target_data, feature_names = self.data_preprocessor.generate_synthetic_medicaid_data()
        
        # Store feature names for later use
        self.feature_names = feature_names
        
        # Save preprocessed data
        self._save_preprocessed_data(source_data, target_data)
        
        logger.info(f"Loaded source data: {source_data[0].shape}, target data: {target_data[0].shape}")
        logger.info(f"Number of features: {len(feature_names)}")
        
        return source_data, target_data
    
    def _save_preprocessed_data(self, source_data: Tuple[np.ndarray, np.ndarray],
                               target_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Save preprocessed data for reproducibility."""
        
        data_dir = self.output_dir / "preprocessed_data"
        data_dir.mkdir(exist_ok=True)
        
        # Save source data
        np.save(data_dir / "X_source.npy", source_data[0])
        np.save(data_dir / "y_source.npy", source_data[1])
        
        # Save target data
        np.save(data_dir / "X_target.npy", target_data[0])
        np.save(data_dir / "y_target.npy", target_data[1])
        
        # Save feature names
        pd.Series(self.feature_names).to_csv(data_dir / "feature_names.csv", index=False)
        
        logger.info(f"Saved preprocessed data to {data_dir}")
    
    def train_all_models(self, source_data: Tuple[np.ndarray, np.ndarray],
                        target_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        """
        Train all transfer learning models.
        
        Args:
            source_data: Source domain data (X_source, y_source)
            target_data: Target domain data (X_target, y_target)
            
        Returns:
            Dictionary of trained models
        """
        logger.info("Training all transfer learning models")
        
        trained_models = {}
        
        for model_name, model_class in self.model_registry.items():
            logger.info(f"Training {model_name}")
            
            try:
                # Initialize model
                model = model_class(self.model_config)
                
                # Train model
                model.fit(source_data, target_data)
                
                # Store trained model
                trained_models[model_name] = model
                
                logger.info(f"Successfully trained {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {str(e)}")
                continue
        
        # Save trained models
        self._save_trained_models(trained_models)
        
        logger.info(f"Successfully trained {len(trained_models)} models")
        
        return trained_models
    
    def _save_trained_models(self, trained_models: Dict[str, Any]) -> None:
        """Save trained models for later use."""
        
        models_dir = self.output_dir / "trained_models"
        models_dir.mkdir(exist_ok=True)
        
        for model_name, model in trained_models.items():
            try:
                # Save model using pickle or joblib
                import joblib
                model_path = models_dir / f"{model_name}.pkl"
                joblib.dump(model, model_path)
                logger.debug(f"Saved {model_name} to {model_path}")
            except Exception as e:
                logger.warning(f"Failed to save {model_name}: {str(e)}")
        
        logger.info(f"Saved trained models to {models_dir}")
    
    def evaluate_models(self, trained_models: Dict[str, Any],
                       source_data: Tuple[np.ndarray, np.ndarray],
                       target_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        """
        Evaluate all trained models.
        
        Args:
            trained_models: Dictionary of trained models
            source_data: Source domain data
            target_data: Target domain data
            
        Returns:
            Dictionary of evaluation results
        """
        logger.info("Evaluating all trained models")
        
        # Comprehensive model evaluation
        evaluation_results = self.evaluation_framework.evaluate_all_models(
            trained_models, source_data, target_data
        )
        
        # Save evaluation results
        self._save_evaluation_results(evaluation_results)
        
        logger.info("Completed model evaluation")
        
        return evaluation_results
    
    def perform_statistical_analysis(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform statistical analysis on evaluation results.
        
        Args:
            evaluation_results: Results from model evaluation
            
        Returns:
            Dictionary of statistical analysis results
        """
        logger.info("Performing statistical analysis")
        
        # Statistical significance testing
        statistical_results = self.statistical_analyzer.perform_statistical_tests(evaluation_results)
        
        # Save statistical results
        self._save_statistical_results(statistical_results)
        
        logger.info("Completed statistical analysis")
        
        return statistical_results
    
    def perform_calibration_analysis(self, trained_models: Dict[str, Any],
                                   target_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        """
        Perform calibration analysis on all models.
        
        Args:
            trained_models: Dictionary of trained models
            target_data: Target domain data
            
        Returns:
            Dictionary of calibration analysis results
        """
        logger.info("Performing calibration analysis")
        
        # Calibration assessment
        calibration_results = self.calibration_analyzer.analyze_calibration(
            trained_models, target_data
        )
        
        # Save calibration results
        self._save_calibration_results(calibration_results)
        
        logger.info("Completed calibration analysis")
        
        return calibration_results
    
    def perform_ablation_study(self, source_data: Tuple[np.ndarray, np.ndarray],
                             target_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        """
        Perform comprehensive ablation study.
        
        Args:
            source_data: Source domain data
            target_data: Target domain data
            
        Returns:
            Dictionary of ablation study results
        """
        logger.info("Performing ablation study")
        
        # Comprehensive ablation analysis
        ablation_results = self.ablation_study.perform_ablation_analysis(
            source_data, target_data, self.feature_names
        )
        
        # Save ablation results
        self._save_ablation_results(ablation_results)
        
        logger.info("Completed ablation study")
        
        return ablation_results
    
    def generate_visualizations(self, evaluation_results: Dict[str, Any],
                              statistical_results: Dict[str, Any],
                              calibration_results: Dict[str, Any],
                              ablation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate all visualization figures.
        
        Args:
            evaluation_results: Model evaluation results
            statistical_results: Statistical analysis results
            calibration_results: Calibration analysis results
            ablation_results: Ablation study results
            
        Returns:
            Dictionary of generated figure paths
        """
        logger.info("Generating visualizations")
        
        # Generate all figures
        visualization_results = self.visualization_generator.generate_all_figures(
            evaluation_results, statistical_results, calibration_results, ablation_results
        )
        
        logger.info("Completed visualization generation")
        
        return visualization_results
    
    def _save_evaluation_results(self, results: Dict[str, Any]) -> None:
        """Save evaluation results to files."""
        
        results_dir = self.output_dir / "evaluation_results"
        results_dir.mkdir(exist_ok=True)
        
        # Save comparison table
        if 'comparison_table' in results:
            results['comparison_table'].to_csv(results_dir / "model_comparison.csv", index=False)
        
        # Save improvement metrics
        if 'improvement_metrics' in results:
            results['improvement_metrics'].to_csv(results_dir / "improvement_metrics.csv", index=False)
        
        logger.info(f"Saved evaluation results to {results_dir}")
    
    def _save_statistical_results(self, results: Dict[str, Any]) -> None:
        """Save statistical analysis results to files."""
        
        results_dir = self.output_dir / "statistical_results"
        results_dir.mkdir(exist_ok=True)
        
        # Save comparison table
        if 'comparison_table' in results:
            results['comparison_table'].to_csv(results_dir / "statistical_comparison.csv", index=False)
        
        # Save significance table
        if 'significance_table' in results:
            results['significance_table'].to_csv(results_dir / "significance_tests.csv", index=False)
        
        logger.info(f"Saved statistical results to {results_dir}")
    
    def _save_calibration_results(self, results: Dict[str, Any]) -> None:
        """Save calibration analysis results to files."""
        
        results_dir = self.output_dir / "calibration_results"
        results_dir.mkdir(exist_ok=True)
        
        # Save comparison table
        if 'comparison_table' in results:
            results['comparison_table'].to_csv(results_dir / "calibration_comparison.csv", index=False)
        
        # Save improvement table
        if 'improvement_table' in results:
            results['improvement_table'].to_csv(results_dir / "calibration_improvement.csv", index=False)
        
        logger.info(f"Saved calibration results to {results_dir}")
    
    def _save_ablation_results(self, results: Dict[str, Any]) -> None:
        """Save ablation study results to files."""
        
        results_dir = self.output_dir / "ablation_results"
        results_dir.mkdir(exist_ok=True)
        
        # Save summary tables
        if 'summary_tables' in results:
            for table_name, table_df in results['summary_tables'].items():
                table_df.to_csv(results_dir / f"{table_name}.csv", index=False)
        
        logger.info(f"Saved ablation results to {results_dir}")
    
    def generate_final_report(self, evaluation_results: Dict[str, Any],
                            statistical_results: Dict[str, Any],
                            calibration_results: Dict[str, Any],
                            ablation_results: Dict[str, Any],
                            visualization_results: Dict[str, Any]) -> None:
        """
        Generate final analysis report.
        
        Args:
            evaluation_results: Model evaluation results
            statistical_results: Statistical analysis results
            calibration_results: Calibration analysis results
            ablation_results: Ablation study results
            visualization_results: Visualization results
        """
        logger.info("Generating final analysis report")
        
        report_path = self.output_dir / "final_analysis_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Medicaid Transfer Learning Analysis Report\n\n")
            
            # Executive summary
            f.write("## Executive Summary\n\n")
            
            best_model = evaluation_results.get('overall_best_model', 'Unknown')
            n_models = len(evaluation_results.get('model_results', {}))
            
            f.write(f"- **Models Evaluated**: {n_models}\n")
            f.write(f"- **Best Performing Model**: {best_model}\n")
            f.write(f"- **Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")
            
            # Model performance summary
            f.write("## Model Performance Summary\n\n")
            
            if 'comparison_table' in evaluation_results:
                comparison_df = evaluation_results['comparison_table']
                
                f.write("### Top 3 Models by AUC:\n")
                top_models = comparison_df.nlargest(3, 'AUC')
                for i, (_, row) in enumerate(top_models.iterrows(), 1):
                    f.write(f"{i}. **{row['Model']}**: AUC = {row['AUC']:.3f}, "
                           f"Youden's J = {row['Youdens_J']:.3f}\n")
                f.write("\n")
            
            # Statistical significance
            f.write("## Statistical Significance\n\n")
            
            if 'model_comparison' in statistical_results:
                n_significant = sum(
                    len([comp for comp in comparisons if comp.get('significant_corrected', False)])
                    for comparisons in statistical_results['model_comparison']['pairwise_comparisons'].values()
                )
                f.write(f"- **Significant Comparisons**: {n_significant}\n")
                f.write(f"- **Multiple Comparison Correction**: {statistical_results['analysis_metadata']['multiple_comparison_method']}\n\n")
            
            # Calibration assessment
            f.write("## Calibration Assessment\n\n")
            
            if 'comparison_table' in calibration_results:
                calibration_df = calibration_results['comparison_table']
                best_calibrated = calibration_df.loc[calibration_df['ECE'].idxmin(), 'Model']
                best_ece = calibration_df['ECE'].min()
                
                f.write(f"- **Best Calibrated Model**: {best_calibrated} (ECE = {best_ece:.3f})\n")
                f.write(f"- **Post-hoc Calibration**: Applied to all models\n\n")
            
            # Feature importance
            f.write("## Feature Importance\n\n")
            
            if 'summary_tables' in ablation_results and 'grouped_importance' in ablation_results['summary_tables']:
                grouped_df = ablation_results['summary_tables']['grouped_importance']
                top_group = grouped_df.loc[grouped_df['Importance_Mean'].idxmax()]
                
                f.write(f"- **Most Important Feature Group**: {top_group['Feature_Group']}\n")
                f.write(f"- **Importance Score**: {top_group['Importance_Mean']:.3f}\n\n")
            
            # Generated outputs
            f.write("## Generated Outputs\n\n")
            
            total_figures = sum(len(figs) for figs in visualization_results.values())
            f.write(f"- **Total Figures Generated**: {total_figures}\n")
            f.write(f"- **Results Directory**: {self.output_dir}\n")
            f.write(f"- **Figures Directory**: {self.output_dir / 'figures'}\n\n")
            
            # Reproducibility information
            f.write("## Reproducibility Information\n\n")
            f.write(f"- **Random Seeds**: {self.model_config.get('random_seeds', {})}\n")
            f.write(f"- **Model Configuration**: {self.model_config.get('models', {}).keys()}\n")
            f.write(f"- **Analysis Pipeline**: Complete\n\n")
            
            f.write("---\n")
            f.write("*Report generated automatically by Medicaid Transfer Learning Analysis Pipeline*\n")
        
        logger.info(f"Generated final report: {report_path}")
    
    def run_complete_analysis(self) -> None:
        """
        Run the complete analysis pipeline.
        """
        logger.info("Starting complete Medicaid transfer learning analysis")
        
        # Set random seeds for reproducibility
        self._set_random_seeds()
        
        # Step 1: Load and preprocess data
        source_data, target_data = self.load_and_preprocess_data()
        
        # Step 2: Train all models
        trained_models = self.train_all_models(source_data, target_data)
        
        # Step 3: Evaluate models
        evaluation_results = self.evaluate_models(trained_models, source_data, target_data)
        
        # Step 4: Statistical analysis
        statistical_results = self.perform_statistical_analysis(evaluation_results)
        
        # Step 5: Calibration analysis
        calibration_results = self.perform_calibration_analysis(trained_models, target_data)
        
        # Step 6: Ablation study
        ablation_results = self.perform_ablation_study(source_data, target_data)
        
        # Step 7: Generate visualizations
        visualization_results = self.generate_visualizations(
            evaluation_results, statistical_results, calibration_results, ablation_results
        )
        
        # Step 8: Generate final report
        self.generate_final_report(
            evaluation_results, statistical_results, calibration_results, 
            ablation_results, visualization_results
        )
        
        logger.info("Completed complete Medicaid transfer learning analysis")
        logger.info(f"All results saved to: {self.output_dir}")


def main():
    """Main function to run the analysis."""
    
    parser = argparse.ArgumentParser(description='Medicaid Transfer Learning Analysis')
    parser.add_argument('--model-config', default='config/model_config.yaml',
                       help='Path to model configuration file')
    parser.add_argument('--data-config', default='config/data_config.yaml',
                       help='Path to data configuration file')
    parser.add_argument('--output-dir', default='results',
                       help='Output directory for results')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        # Initialize and run analysis
        analysis = MedicaidTransferLearningAnalysis(
            args.model_config, args.data_config, args.output_dir
        )
        
        analysis.run_complete_analysis()
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"📁 Results saved to: {args.output_dir}")
        print(f"📊 Check the final report: {args.output_dir}/final_analysis_report.md")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()

