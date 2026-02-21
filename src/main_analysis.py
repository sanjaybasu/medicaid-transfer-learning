#!/usr/bin/env python3
"""
Main Analysis Pipeline for Medicaid Transfer Learning Study

This script orchestrates the complete analysis pipeline including:
- Data preprocessing and quality checks
- Transfer learning model training and evaluation
- Statistical analysis and significance testing
- Calibration analysis and post-hoc correction
- Ablation studies and feature importance
- Visualization and results generation

Usage:
    python main_analysis.py [--config CONFIG_PATH] [--output OUTPUT_DIR]
"""

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Import local modules
from data_preprocessing import DataPreprocessor
from transfer_learning_models import TransferLearningPipeline
from evaluation_metrics import EvaluationFramework
from statistical_analysis import StatisticalAnalyzer
from calibration_analysis import CalibrationAnalyzer
from ablation_study import AblationStudy
from visualization import VisualizationGenerator
from fairness_analysis import FairnessAnalyzer
from component_outcome_analysis import ComponentOutcomeAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class MedicaidTransferLearningAnalysis:
    """
    Main analysis class that orchestrates the complete transfer learning pipeline.
    """
    
    def __init__(self, config_path: str = "config/", output_dir: str = "results/"):
        """
        Initialize the analysis pipeline.
        
        Args:
            config_path: Path to configuration files
            output_dir: Directory for output files
        """
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configurations
        self.data_config = self._load_config("data_config.yaml")
        self.model_config = self._load_config("model_config.yaml")
        
        # Initialize components
        self.data_preprocessor = DataPreprocessor(self.data_config)
        self.model_pipeline = TransferLearningPipeline(self.model_config)
        self.evaluator = EvaluationFramework(self.model_config)
        self.statistical_analyzer = StatisticalAnalyzer(self.model_config)
        self.calibration_analyzer = CalibrationAnalyzer(self.model_config)
        self.ablation_study = AblationStudy(self.model_config)
        self.visualizer = VisualizationGenerator(self.output_dir)
        self.fairness_analyzer = FairnessAnalyzer(self.model_config)
        self.component_analyzer = ComponentOutcomeAnalyzer(self.model_config)
        
        # Set random seeds for reproducibility
        self._set_random_seeds()
        
    def _load_config(self, config_file: str) -> dict:
        """Load configuration from YAML file."""
        config_path = self.config_path / config_file
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _set_random_seeds(self):
        """Set random seeds for reproducibility."""
        seeds = self.model_config['random_seeds']
        np.random.seed(seeds['model_init'])
        
    def run_complete_analysis(self):
        """
        Execute the complete analysis pipeline.
        """
        logger.info("Starting Medicaid Transfer Learning Analysis")
        start_time = datetime.now()
        
        try:
            # Step 1: Data preprocessing and quality checks
            logger.info("Step 1: Data preprocessing and quality checks")
            source_data, target_data = self.data_preprocessor.load_and_preprocess()
            
            # Step 2: Train transfer learning models
            logger.info("Step 2: Training transfer learning models")
            trained_models = self.model_pipeline.train_all_models(source_data, target_data)
            
            # Step 3: Evaluate model performance
            logger.info("Step 3: Evaluating model performance")
            performance_results = self.evaluator.evaluate_all_models(
                trained_models, source_data, target_data
            )
            
            # Step 4: Statistical analysis and significance testing
            logger.info("Step 4: Statistical analysis and significance testing")
            statistical_results = self.statistical_analyzer.perform_statistical_tests(
                performance_results
            )

            # Step 4b: Fairness analysis (equalized odds across demographic subgroups)
            logger.info("Step 4b: Fairness analysis")
            predictions_df = performance_results.get("predictions")
            fairness_results: dict = {}
            if predictions_df is not None and not predictions_df.empty:
                model_thresholds = None
                if "model_comparison" in performance_results:
                    mc = performance_results["model_comparison"]
                    if "OptimalThreshold" in mc.columns:
                        model_thresholds = dict(zip(mc["Model"], mc["OptimalThreshold"]))
                fairness_results = self.fairness_analyzer.analyze_fairness(
                    predictions=predictions_df,
                    target_demographics=target_data.test_demographics,
                    model_thresholds=model_thresholds,
                ) if target_data.test_demographics is not None else {}
                if not fairness_results:
                    logger.warning("Fairness analysis skipped: test_demographics not available.")
            else:
                logger.warning("Fairness analysis skipped: no predictions DataFrame.")

            # Step 4c: Component outcome analysis (ED-only and hospitalisation-only)
            logger.info("Step 4c: Component outcome analysis")
            component_results: dict = {}
            if target_data.raw_df is not None and target_data.test_indices is not None:
                component_results = self.component_analyzer.analyze(
                    trained_models=trained_models,
                    raw_target_df=target_data.raw_df,
                    test_indices=target_data.test_indices,
                    target_X_test=target_data.X_test,
                )
            else:
                logger.warning("Component outcome analysis skipped: raw_df or test_indices not available.")

            # Step 5: Calibration analysis
            logger.info("Step 5: Calibration analysis")
            calibration_results = self.calibration_analyzer.analyze_calibration(
                trained_models, target_data
            )
            
            # Step 6: Ablation studies
            logger.info("Step 6: Ablation studies")
            ablation_results = self.ablation_study.perform_ablation_analysis(
                source_data,
                target_data,
                trained_models,
                performance_results.get("model_comparison") if performance_results else None,
            )
            
            # Step 7: Generate visualizations
            logger.info("Step 7: Generating visualizations")
            self.visualizer.generate_all_figures(
                performance_results,
                statistical_results,
                calibration_results,
                ablation_results
            )
            
            # Step 8: Save results
            logger.info("Step 8: Saving results")
            self._save_results({
                'performance': performance_results,
                'statistical': statistical_results,
                'calibration': calibration_results,
                'ablation': ablation_results,
                'fairness': fairness_results,
                'component': component_results,
            })
            
            # Generate summary report
            self._generate_summary_report(performance_results, statistical_results)
            
            end_time = datetime.now()
            logger.info(f"Analysis completed successfully in {end_time - start_time}")
            
        except Exception as e:
            logger.error(f"Analysis failed with error: {str(e)}")
            raise
    
    def _save_results(self, results: dict):
        """Save all results to files."""
        results_dir = self.output_dir / "tables"
        results_dir.mkdir(exist_ok=True)
        
        for category, data in results.items():
            if isinstance(data, dict):
                for name, df in data.items():
                    if isinstance(df, pd.DataFrame):
                        filename = f"{category}_{name}.csv"
                        df.to_csv(results_dir / filename, index=False)
                        logger.info(f"Saved {filename}")
    
    def _generate_summary_report(self, performance_results: dict, statistical_results: dict):
        """Generate a summary report of key findings."""
        report_path = self.output_dir / "analysis_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("MEDICAID TRANSFER LEARNING ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Best performing model
            if 'model_comparison' in performance_results:
                best_model = performance_results['model_comparison'].loc[
                    performance_results['model_comparison']['AUC'].idxmax()
                ]
                f.write(f"Best performing model: {best_model['Model']}\n")
                f.write(f"AUC: {best_model['AUC']:.3f}\n")
                f.write(f"Youden's J: {best_model['YoudensJ']:.3f}\n")
                f.write(f"Sensitivity: {best_model['Sensitivity']:.3f}\n")
                f.write(f"Specificity: {best_model['Specificity']:.3f}\n")
                f.write(f"Precision: {best_model['Precision']:.3f}\n\n")
            
            # Statistical significance
            if 'significance_tests' in statistical_results:
                sig_tests = statistical_results['significance_tests']
                significant_improvements = sig_tests[sig_tests['p_value'] < 0.05]
                f.write(f"Statistically significant improvements: {len(significant_improvements)}\n")
                for _, row in significant_improvements.iterrows():
                    f.write(f"  {row['Comparison']}: p = {row['p_value']:.3f}\n")
            
            f.write("\nFor detailed results, see the tables/ and figures/ directories.\n")
        
        logger.info(f"Summary report saved to {report_path}")


def main():
    """Main entry point for the analysis."""
    parser = argparse.ArgumentParser(
        description="Run Medicaid Transfer Learning Analysis"
    )
    parser.add_argument(
        "--config", 
        default="config/",
        help="Path to configuration directory"
    )
    parser.add_argument(
        "--output",
        default="results/",
        help="Output directory for results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize and run analysis
    analysis = MedicaidTransferLearningAnalysis(
        config_path=args.config,
        output_dir=args.output
    )
    
    analysis.run_complete_analysis()


if __name__ == "__main__":
    main()
