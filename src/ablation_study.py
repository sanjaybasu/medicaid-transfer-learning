#!/usr/bin/env python3
"""
Ablation Study Module for Medicaid Transfer Learning Study

This module provides comprehensive ablation study capabilities including
feature importance analysis, component ablation, and systematic feature removal.

Classes:
    AblationStudy: Main ablation study framework
    FeatureImportanceAnalyzer: Feature importance calculation utilities
    ComponentAblation: Model component ablation methods
    PermutationImportance: Permutation-based feature importance
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class PermutationImportance:
    """
    Permutation-based feature importance calculation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize permutation importance calculator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.ablation_config = config.get('ablation', {})
        self.n_repeats = self.ablation_config.get('n_repeats', 10)
        self.random_state = config.get('random_seeds', {}).get('ablation', 42)
    
    def calculate_permutation_importance(self, model, X: np.ndarray, y: np.ndarray, 
                                       scoring: str = 'roc_auc') -> Dict[str, Any]:
        """
        Calculate permutation importance for all features.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target labels
            scoring: Scoring metric
            
        Returns:
            Dictionary with importance scores and statistics
        """
        logger.info("Calculating permutation importance")
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            model, X, y, 
            n_repeats=self.n_repeats,
            random_state=self.random_state,
            scoring=scoring
        )
        
        # Calculate statistics
        importance_stats = {
            'importances_mean': perm_importance.importances_mean,
            'importances_std': perm_importance.importances_std,
            'importances': perm_importance.importances,
        }
        
        # Rank features by importance
        feature_ranking = np.argsort(perm_importance.importances_mean)[::-1]
        
        results = {
            'importance_stats': importance_stats,
            'feature_ranking': feature_ranking,
            'top_features': feature_ranking[:20],  # Top 20 features
            'scoring_metric': scoring
        }
        
        logger.info("Completed permutation importance calculation")
        
        return results
    
    def calculate_grouped_importance(self, model, X: np.ndarray, y: np.ndarray,
                                   feature_groups: Dict[str, List[int]], 
                                   scoring: str = 'roc_auc') -> Dict[str, Any]:
        """
        Calculate permutation importance for feature groups.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target labels
            feature_groups: Dictionary mapping group names to feature indices
            scoring: Scoring metric
            
        Returns:
            Dictionary with group importance scores
        """
        logger.info("Calculating grouped permutation importance")
        
        # Baseline score
        baseline_score = self._calculate_score(model, X, y, scoring)
        
        group_importances = {}
        
        for group_name, feature_indices in feature_groups.items():
            logger.debug(f"Calculating importance for group: {group_name}")
            
            group_scores = []
            
            for repeat in range(self.n_repeats):
                # Create permuted data
                X_permuted = X.copy()
                
                # Permute all features in the group
                for feature_idx in feature_indices:
                    if feature_idx < X.shape[1]:
                        np.random.shuffle(X_permuted[:, feature_idx])
                
                # Calculate score with permuted features
                permuted_score = self._calculate_score(model, X_permuted, y, scoring)
                group_scores.append(baseline_score - permuted_score)
            
            group_importances[group_name] = {
                'importance_mean': np.mean(group_scores),
                'importance_std': np.std(group_scores),
                'importance_scores': group_scores,
                'feature_indices': feature_indices,
                'n_features': len(feature_indices)
            }
        
        logger.info("Completed grouped permutation importance calculation")
        
        return {
            'baseline_score': baseline_score,
            'group_importances': group_importances,
            'scoring_metric': scoring
        }
    
    def _calculate_score(self, model, X: np.ndarray, y: np.ndarray, scoring: str) -> float:
        """Calculate score for given model and data."""
        y_pred = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X)
        
        if scoring == 'roc_auc':
            return roc_auc_score(y, y_pred)
        elif scoring == 'f1':
            y_pred_binary = (y_pred > 0.5).astype(int) if y_pred.dtype == float else y_pred
            return f1_score(y, y_pred_binary)
        elif scoring == 'mcc':
            y_pred_binary = (y_pred > 0.5).astype(int) if y_pred.dtype == float else y_pred
            return matthews_corrcoef(y, y_pred_binary)
        else:
            raise ValueError(f"Unsupported scoring metric: {scoring}")


class FeatureImportanceAnalyzer:
    """
    Feature importance calculation utilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature importance analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.ablation_config = config.get('ablation', {})
    
    def calculate_model_specific_importance(self, model, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate model-specific feature importance.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            
        Returns:
            Dictionary with importance scores
        """
        importance_scores = None
        importance_type = 'unknown'
        
        # Try different model-specific importance methods
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importance_scores = model.feature_importances_
            importance_type = 'tree_importance'
            
        elif hasattr(model, 'coef_'):
            # Linear models
            importance_scores = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
            importance_type = 'linear_coefficients'
            
        elif hasattr(model, 'model') and hasattr(model.model, 'coef_'):
            # Wrapped models (e.g., in transfer learning classes)
            coef = model.model.coef_
            importance_scores = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)
            importance_type = 'linear_coefficients'
        
        if importance_scores is not None:
            # Normalize importance scores
            importance_scores = importance_scores / np.sum(importance_scores)
            
            # Rank features
            feature_ranking = np.argsort(importance_scores)[::-1]
            
            results = {
                'importance_scores': importance_scores,
                'importance_type': importance_type,
                'feature_ranking': feature_ranking,
                'top_features': feature_ranking[:20],
                'feature_names': feature_names
            }
            
            if feature_names:
                results['top_feature_names'] = [feature_names[i] for i in feature_ranking[:20]]
            
            return results
        
        else:
            logger.warning("Model does not support intrinsic feature importance")
            return None
    
    def calculate_drop_column_importance(self, model_class, model_params: Dict[str, Any],
                                       X: np.ndarray, y: np.ndarray,
                                       feature_names: Optional[List[str]] = None,
                                       scoring: str = 'roc_auc') -> Dict[str, Any]:
        """
        Calculate feature importance using drop-column method.
        
        Args:
            model_class: Model class to instantiate
            model_params: Model parameters
            X: Feature matrix
            y: Target labels
            feature_names: List of feature names
            scoring: Scoring metric
            
        Returns:
            Dictionary with drop-column importance scores
        """
        logger.info("Calculating drop-column importance")
        
        # Baseline score with all features
        baseline_model = model_class(**model_params)
        baseline_model.fit(X, y)
        baseline_score = self._calculate_score(baseline_model, X, y, scoring)
        
        importance_scores = []
        
        for feature_idx in range(X.shape[1]):
            logger.debug(f"Dropping feature {feature_idx}")
            
            # Create dataset without this feature
            X_dropped = np.delete(X, feature_idx, axis=1)
            
            # Train model without this feature
            dropped_model = model_class(**model_params)
            dropped_model.fit(X_dropped, y)
            
            # Calculate score without this feature
            dropped_score = self._calculate_score(dropped_model, X_dropped, y, scoring)
            
            # Importance is the drop in performance
            importance = baseline_score - dropped_score
            importance_scores.append(importance)
        
        importance_scores = np.array(importance_scores)
        
        # Rank features by importance
        feature_ranking = np.argsort(importance_scores)[::-1]
        
        results = {
            'baseline_score': baseline_score,
            'importance_scores': importance_scores,
            'feature_ranking': feature_ranking,
            'top_features': feature_ranking[:20],
            'scoring_metric': scoring,
            'feature_names': feature_names
        }
        
        if feature_names:
            results['top_feature_names'] = [feature_names[i] for i in feature_ranking[:20]]
        
        logger.info("Completed drop-column importance calculation")
        
        return results
    
    def _calculate_score(self, model, X: np.ndarray, y: np.ndarray, scoring: str) -> float:
        """Calculate score for given model and data."""
        y_pred = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X)
        
        if scoring == 'roc_auc':
            return roc_auc_score(y, y_pred)
        elif scoring == 'f1':
            y_pred_binary = (y_pred > 0.5).astype(int) if y_pred.dtype == float else y_pred
            return f1_score(y, y_pred_binary)
        elif scoring == 'mcc':
            y_pred_binary = (y_pred > 0.5).astype(int) if y_pred.dtype == float else y_pred
            return matthews_corrcoef(y, y_pred_binary)
        else:
            raise ValueError(f"Unsupported scoring metric: {scoring}")


class ComponentAblation:
    """
    Model component ablation methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize component ablation.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.ablation_config = config.get('ablation', {})
    
    def ablate_transfer_learning_components(self, model_class, base_params: Dict[str, Any],
                                          source_data: Tuple[np.ndarray, np.ndarray],
                                          target_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        """
        Ablate transfer learning components.
        
        Args:
            model_class: Transfer learning model class
            base_params: Base model parameters
            source_data: Source domain data
            target_data: Target domain data
            
        Returns:
            Dictionary with ablation results
        """
        logger.info("Performing transfer learning component ablation")
        
        X_target, y_target = target_data
        
        # Define ablation configurations
        ablation_configs = {
            'full_model': base_params,
            'no_domain_adaptation': {**base_params, 'lambda_domain': 0.0},
            'no_meta_learning': {**base_params, 'n_inner_steps': 0},
            'reduced_embedding': {**base_params, 'embedding_dim': base_params.get('embedding_dim', 64) // 2},
            'no_regularization': {**base_params, 'dropout': 0.0, 'weight_decay': 0.0}
        }
        
        ablation_results = {}
        
        for config_name, config_params in ablation_configs.items():
            logger.debug(f"Testing configuration: {config_name}")
            
            try:
                # Train model with ablated configuration
                model = model_class(self.config)
                
                # Update model parameters if applicable
                for param, value in config_params.items():
                    if hasattr(model, param):
                        setattr(model, param, value)
                
                model.fit(source_data, target_data)
                
                # Evaluate performance
                y_pred_proba = model.predict_proba(X_target)[:, 1]
                
                performance = {
                    'auc_roc': roc_auc_score(y_target, y_pred_proba),
                    'f1_score': f1_score(y_target, (y_pred_proba > 0.5).astype(int)),
                    'mcc': matthews_corrcoef(y_target, (y_pred_proba > 0.5).astype(int))
                }
                
                ablation_results[config_name] = {
                    'config': config_params,
                    'performance': performance
                }
                
            except Exception as e:
                logger.warning(f"Failed to test configuration {config_name}: {str(e)}")
                continue
        
        # Calculate performance drops
        if 'full_model' in ablation_results:
            baseline_performance = ablation_results['full_model']['performance']
            
            for config_name, results in ablation_results.items():
                if config_name != 'full_model':
                    performance_drop = {}
                    for metric, value in results['performance'].items():
                        baseline_value = baseline_performance[metric]
                        drop = baseline_value - value
                        drop_percent = (drop / baseline_value) * 100 if baseline_value != 0 else 0
                        performance_drop[f'{metric}_drop'] = drop
                        performance_drop[f'{metric}_drop_percent'] = drop_percent
                    
                    results['performance_drop'] = performance_drop
        
        logger.info("Completed transfer learning component ablation")
        
        return ablation_results


class AblationStudy:
    """
    Main ablation study framework.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ablation study framework.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.permutation_analyzer = PermutationImportance(config)
        self.feature_analyzer = FeatureImportanceAnalyzer(config)
        self.component_ablation = ComponentAblation(config)
        
        # Feature groups from config
        self.feature_groups = config.get('ablation', {}).get('feature_groups', {})
    
    def perform_feature_importance_analysis(self, model, X: np.ndarray, y: np.ndarray,
                                          feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive feature importance analysis.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target labels
            feature_names: List of feature names
            
        Returns:
            Dictionary with feature importance results
        """
        logger.info("Performing comprehensive feature importance analysis")
        
        results = {}
        
        # Model-specific importance
        model_importance = self.feature_analyzer.calculate_model_specific_importance(
            model, feature_names
        )
        if model_importance:
            results['model_specific'] = model_importance
        
        # Permutation importance
        perm_importance = self.permutation_analyzer.calculate_permutation_importance(
            model, X, y, scoring='roc_auc'
        )
        results['permutation'] = perm_importance
        
        # Grouped permutation importance
        if self.feature_groups and feature_names:
            # Map feature names to indices
            feature_name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
            
            mapped_feature_groups = {}
            for group_name, group_features in self.feature_groups.items():
                group_indices = []
                for feature_name in group_features:
                    if feature_name in feature_name_to_idx:
                        group_indices.append(feature_name_to_idx[feature_name])
                
                if group_indices:
                    mapped_feature_groups[group_name] = group_indices
            
            if mapped_feature_groups:
                grouped_importance = self.permutation_analyzer.calculate_grouped_importance(
                    model, X, y, mapped_feature_groups, scoring='roc_auc'
                )
                results['grouped_permutation'] = grouped_importance
        
        logger.info("Completed comprehensive feature importance analysis")
        
        return results
    
    def perform_ablation_analysis(self, source_data: Tuple[np.ndarray, np.ndarray],
                                target_data: Tuple[np.ndarray, np.ndarray],
                                feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive ablation analysis.
        
        Args:
            source_data: Source domain data
            target_data: Target domain data
            feature_names: List of feature names
            
        Returns:
            Dictionary with ablation analysis results
        """
        logger.info("Performing comprehensive ablation analysis")
        
        X_target, y_target = target_data
        
        # Import transfer learning models for ablation
        from transfer_learning_models import (
            SourceOnlyTransfer, MAML, DomainAdversarialNetwork, 
            PrototypicalNetworks, TabTransformer, MetaEnsemble
        )
        
        ablation_results = {}
        
        # Feature importance analysis for each model type
        model_classes = {
            'source_only': SourceOnlyTransfer,
            'maml': MAML,
            'domain_adversarial': DomainAdversarialNetwork,
            'prototypical_networks': PrototypicalNetworks,
            'tab_transformer': TabTransformer,
            'meta_ensemble': MetaEnsemble
        }
        
        for model_name, model_class in model_classes.items():
            logger.info(f"Performing ablation for {model_name}")
            
            try:
                # Train model
                model = model_class(self.config)
                model.fit(source_data, target_data)
                
                # Feature importance analysis
                feature_importance = self.perform_feature_importance_analysis(
                    model, X_target, y_target, feature_names
                )
                
                # Component ablation (for complex models)
                if model_name in ['maml', 'domain_adversarial', 'tab_transformer']:
                    component_ablation = self.component_ablation.ablate_transfer_learning_components(
                        model_class, self.config['models'][model_name], source_data, target_data
                    )
                    feature_importance['component_ablation'] = component_ablation
                
                ablation_results[model_name] = feature_importance
                
            except Exception as e:
                logger.error(f"Failed to perform ablation for {model_name}: {str(e)}")
                continue
        
        # Generate summary tables
        summary_tables = self._generate_ablation_summary_tables(ablation_results)
        
        final_results = {
            'model_ablation_results': ablation_results,
            'summary_tables': summary_tables,
            'feature_groups': self.feature_groups,
            'analysis_metadata': {
                'n_models_analyzed': len(ablation_results),
                'target_domain_size': len(y_target),
                'feature_names': feature_names,
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            }
        }
        
        logger.info("Completed comprehensive ablation analysis")
        
        return final_results
    
    def _generate_ablation_summary_tables(self, ablation_results: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Generate summary tables for ablation results."""
        
        summary_tables = {}
        
        # Feature importance summary
        feature_importance_data = []
        
        for model_name, results in ablation_results.items():
            if 'permutation' in results:
                perm_results = results['permutation']
                importance_stats = perm_results['importance_stats']
                
                for i, (mean_imp, std_imp) in enumerate(zip(
                    importance_stats['importances_mean'], 
                    importance_stats['importances_std']
                )):
                    feature_importance_data.append({
                        'Model': model_name,
                        'Feature_Index': i,
                        'Importance_Mean': mean_imp,
                        'Importance_Std': std_imp,
                        'Importance_CV': std_imp / mean_imp if mean_imp != 0 else 0
                    })
        
        if feature_importance_data:
            summary_tables['feature_importance'] = pd.DataFrame(feature_importance_data)
        
        # Grouped importance summary
        grouped_importance_data = []
        
        for model_name, results in ablation_results.items():
            if 'grouped_permutation' in results:
                grouped_results = results['grouped_permutation']
                
                for group_name, group_data in grouped_results['group_importances'].items():
                    grouped_importance_data.append({
                        'Model': model_name,
                        'Feature_Group': group_name,
                        'Importance_Mean': group_data['importance_mean'],
                        'Importance_Std': group_data['importance_std'],
                        'N_Features': group_data['n_features']
                    })
        
        if grouped_importance_data:
            summary_tables['grouped_importance'] = pd.DataFrame(grouped_importance_data)
        
        # Component ablation summary
        component_ablation_data = []
        
        for model_name, results in ablation_results.items():
            if 'component_ablation' in results:
                comp_results = results['component_ablation']
                
                for config_name, config_data in comp_results.items():
                    if 'performance_drop' in config_data:
                        performance_drop = config_data['performance_drop']
                        
                        component_ablation_data.append({
                            'Model': model_name,
                            'Component_Removed': config_name,
                            'AUC_Drop': performance_drop.get('auc_roc_drop', 0),
                            'AUC_Drop_Percent': performance_drop.get('auc_roc_drop_percent', 0),
                            'F1_Drop': performance_drop.get('f1_score_drop', 0),
                            'F1_Drop_Percent': performance_drop.get('f1_score_drop_percent', 0),
                            'MCC_Drop': performance_drop.get('mcc_drop', 0),
                            'MCC_Drop_Percent': performance_drop.get('mcc_drop_percent', 0)
                        })
        
        if component_ablation_data:
            summary_tables['component_ablation'] = pd.DataFrame(component_ablation_data)
        
        return summary_tables

