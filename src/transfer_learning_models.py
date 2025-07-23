#!/usr/bin/env python3
"""
Transfer Learning Models for Medicaid Risk Prediction

This module implements various transfer learning approaches for predicting
acute care utilization across different Medicaid populations.

Classes:
    BaseTransferModel: Abstract base class for transfer learning models
    SourceOnlyTransfer: Naive transfer learning baseline
    PrototypicalNetworks: Few-shot learning with prototypical networks
    MAML: Model-Agnostic Meta-Learning implementation
    DomainAdversarialNetwork: Domain adversarial training
    CausalTransferLearning: Causal inference-based transfer
    TabTransformer: Transformer architecture for tabular data
    MetaEnsemble: Ensemble of transfer learning approaches
    TransferLearningPipeline: Main pipeline for training all models
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from abc import ABC, abstractmethod
import logging
from typing import Dict, List, Tuple, Optional, Any
import math

logger = logging.getLogger(__name__)


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer for Domain Adversarial Training.
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_
        return output, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer module.
    """
    def __init__(self, lambda_=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class BaseTransferModel(ABC):
    """
    Abstract base class for transfer learning models.
    
    All transfer learning implementations should inherit from this class
    and implement the required abstract methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the transfer learning model.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, source_data: Tuple[np.ndarray, np.ndarray], 
            target_data: Tuple[np.ndarray, np.ndarray]) -> 'BaseTransferModel':
        """
        Fit the transfer learning model.
        
        Args:
            source_data: Tuple of (X_source, y_source)
            target_data: Tuple of (X_target, y_target)
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities
        """
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Input features
            
        Returns:
            Predicted class labels
        """
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] > 0.5).astype(int)


class SourceOnlyTransfer(BaseTransferModel):
    """
    Source-Only Transfer Learning (Naive Transfer).
    
    This baseline approach trains a model on the source domain and applies
    it directly to the target domain without any adaptation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        model_params = config['models']['source_only']
        self.model = LogisticRegression(
            C=model_params['C'],
            max_iter=model_params['max_iter'],
            solver=model_params['solver'],
            random_state=config['random_seeds']['model_init']
        )
    
    def fit(self, source_data: Tuple[np.ndarray, np.ndarray], 
            target_data: Tuple[np.ndarray, np.ndarray]) -> 'SourceOnlyTransfer':
        """
        Fit the model using only source domain data.
        
        Args:
            source_data: Tuple of (X_source, y_source)
            target_data: Tuple of (X_target, y_target) - not used
            
        Returns:
            Self for method chaining
        """
        X_source, y_source = source_data
        
        # Standardize features using source domain statistics
        X_source_scaled = self.scaler.fit_transform(X_source)
        
        # Train model on source domain
        self.model.fit(X_source_scaled, y_source)
        self.is_fitted = True
        
        logger.info("Source-Only Transfer model fitted successfully")
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using source domain model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class PrototypicalNetworks(BaseTransferModel):
    """
    Prototypical Networks for Few-Shot Domain Adaptation.
    
    This approach learns representative prototypes for each class and domain,
    enabling few-shot adaptation to new domains.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.params = config['models']['prototypical_networks']
        self.embedding_dim = self.params['embedding_dim']
        self.n_support = self.params['n_support']
        self.distance_metric = self.params['distance_metric']
        
        # Initialize embedding network
        self.embedding_net = None
        self.prototypes = {}
        
    def _build_embedding_network(self, input_dim: int) -> nn.Module:
        """Build the embedding network."""
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.embedding_dim)
        )
    
    def _compute_prototypes(self, embeddings: torch.Tensor, 
                          labels: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Compute class prototypes from embeddings."""
        prototypes = {}
        for class_label in torch.unique(labels):
            class_mask = labels == class_label
            class_embeddings = embeddings[class_mask]
            prototypes[class_label.item()] = class_embeddings.mean(dim=0)
        return prototypes
    
    def _euclidean_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Euclidean distance between tensors."""
        return torch.sqrt(torch.sum((x - y) ** 2, dim=-1))
    
    def fit(self, source_data: Tuple[np.ndarray, np.ndarray], 
            target_data: Tuple[np.ndarray, np.ndarray]) -> 'PrototypicalNetworks':
        """
        Fit the prototypical networks model.
        
        Args:
            source_data: Tuple of (X_source, y_source)
            target_data: Tuple of (X_target, y_target)
            
        Returns:
            Self for method chaining
        """
        X_source, y_source = source_data
        X_target, y_target = target_data
        
        # Standardize features
        X_source_scaled = self.scaler.fit_transform(X_source)
        X_target_scaled = self.scaler.transform(X_target)
        
        # Convert to tensors
        X_source_tensor = torch.FloatTensor(X_source_scaled)
        y_source_tensor = torch.LongTensor(y_source)
        X_target_tensor = torch.FloatTensor(X_target_scaled)
        y_target_tensor = torch.LongTensor(y_target)
        
        # Build embedding network
        input_dim = X_source.shape[1]
        self.embedding_net = self._build_embedding_network(input_dim)
        
        # Training loop
        optimizer = optim.Adam(self.embedding_net.parameters(), 
                             lr=self.params['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(self.params['n_epochs']):
            # Generate embeddings
            source_embeddings = self.embedding_net(X_source_tensor)
            target_embeddings = self.embedding_net(X_target_tensor)
            
            # Compute prototypes
            source_prototypes = self._compute_prototypes(source_embeddings, y_source_tensor)
            
            # Few-shot adaptation on target domain
            # Use small support set from target domain
            n_support_target = min(self.n_support, len(X_target) // 2)
            support_indices = np.random.choice(len(X_target), n_support_target, replace=False)
            query_indices = np.setdiff1d(np.arange(len(X_target)), support_indices)
            
            support_embeddings = target_embeddings[support_indices]
            support_labels = y_target_tensor[support_indices]
            query_embeddings = target_embeddings[query_indices]
            query_labels = y_target_tensor[query_indices]
            
            # Update prototypes with target support set
            target_prototypes = self._compute_prototypes(support_embeddings, support_labels)
            
            # Compute distances and predictions for query set
            distances = []
            for class_label in [0, 1]:
                if class_label in target_prototypes:
                    prototype = target_prototypes[class_label]
                else:
                    prototype = source_prototypes[class_label]
                
                class_distances = self._euclidean_distance(query_embeddings, prototype)
                distances.append(class_distances)
            
            distances = torch.stack(distances, dim=1)
            predictions = -distances  # Negative distance as logits
            
            # Compute loss
            loss = criterion(predictions, query_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                logger.debug(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Store final prototypes
        with torch.no_grad():
            final_source_embeddings = self.embedding_net(X_source_tensor)
            final_target_embeddings = self.embedding_net(X_target_tensor)
            
            self.prototypes['source'] = self._compute_prototypes(
                final_source_embeddings, y_source_tensor
            )
            self.prototypes['target'] = self._compute_prototypes(
                final_target_embeddings, y_target_tensor
            )
        
        self.is_fitted = True
        logger.info("Prototypical Networks model fitted successfully")
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using prototypical networks.
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        with torch.no_grad():
            embeddings = self.embedding_net(X_tensor)
            
            # Use target prototypes if available, otherwise source prototypes
            prototypes = self.prototypes.get('target', self.prototypes['source'])
            
            distances = []
            for class_label in [0, 1]:
                prototype = prototypes[class_label]
                class_distances = self._euclidean_distance(embeddings, prototype)
                distances.append(class_distances)
            
            distances = torch.stack(distances, dim=1)
            probabilities = torch.softmax(-distances, dim=1)
            
        return probabilities.numpy()


class MAML(BaseTransferModel):
    """
    Model-Agnostic Meta-Learning (MAML) for Transfer Learning.
    
    MAML learns initial parameters that can be quickly adapted to new tasks
    with few gradient steps.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.params = config['models']['maml']
        self.inner_lr = self.params['inner_lr']
        self.outer_lr = self.params['outer_lr']
        self.n_inner_steps = self.params['n_inner_steps']
        self.first_order = self.params['first_order']
        
        self.meta_model = None
        self.adapted_model = None
        
    def _build_model(self, input_dim: int) -> nn.Module:
        """Build the neural network model."""
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)  # Binary classification
        )
    
    def _inner_loop_update(self, model: nn.Module, support_x: torch.Tensor, 
                          support_y: torch.Tensor) -> nn.Module:
        """Perform inner loop adaptation."""
        # Clone model for adaptation
        adapted_model = type(model)(model.in_features if hasattr(model, 'in_features') else None)
        adapted_model.load_state_dict(model.state_dict())
        
        criterion = nn.CrossEntropyLoss()
        
        for step in range(self.n_inner_steps):
            # Forward pass
            logits = adapted_model(support_x)
            loss = criterion(logits, support_y)
            
            # Compute gradients
            grads = torch.autograd.grad(
                loss, adapted_model.parameters(), 
                create_graph=not self.first_order
            )
            
            # Update parameters
            with torch.no_grad():
                for param, grad in zip(adapted_model.parameters(), grads):
                    param.data = param.data - self.inner_lr * grad
        
        return adapted_model
    
    def fit(self, source_data: Tuple[np.ndarray, np.ndarray], 
            target_data: Tuple[np.ndarray, np.ndarray]) -> 'MAML':
        """
        Fit the MAML model.
        
        Args:
            source_data: Tuple of (X_source, y_source)
            target_data: Tuple of (X_target, y_target)
            
        Returns:
            Self for method chaining
        """
        X_source, y_source = source_data
        X_target, y_target = target_data
        
        # Standardize features
        X_source_scaled = self.scaler.fit_transform(X_source)
        X_target_scaled = self.scaler.transform(X_target)
        
        # Convert to tensors
        X_source_tensor = torch.FloatTensor(X_source_scaled)
        y_source_tensor = torch.LongTensor(y_source)
        X_target_tensor = torch.FloatTensor(X_target_scaled)
        y_target_tensor = torch.LongTensor(y_target)
        
        # Build meta-model
        input_dim = X_source.shape[1]
        self.meta_model = self._build_model(input_dim)
        
        # Meta-optimizer
        meta_optimizer = optim.Adam(self.meta_model.parameters(), lr=self.outer_lr)
        criterion = nn.CrossEntropyLoss()
        
        # Split target data into support and query sets
        split_idx = int(len(X_target) * self.params['support_query_split'])
        support_x = X_target_tensor[:split_idx]
        support_y = y_target_tensor[:split_idx]
        query_x = X_target_tensor[split_idx:]
        query_y = y_target_tensor[split_idx:]
        
        for epoch in range(self.params['n_epochs']):
            # Meta-training step
            meta_optimizer.zero_grad()
            
            # Inner loop: adapt to target support set
            adapted_model = self._inner_loop_update(self.meta_model, support_x, support_y)
            
            # Outer loop: evaluate on target query set
            query_logits = adapted_model(query_x)
            meta_loss = criterion(query_logits, query_y)
            
            # Meta-gradient step
            meta_loss.backward()
            meta_optimizer.step()
            
            if epoch % 20 == 0:
                logger.debug(f"MAML Epoch {epoch}, Meta Loss: {meta_loss.item():.4f}")
        
        # Final adaptation for inference
        self.adapted_model = self._inner_loop_update(self.meta_model, support_x, support_y)
        
        self.is_fitted = True
        logger.info("MAML model fitted successfully")
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using adapted MAML model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        with torch.no_grad():
            logits = self.adapted_model(X_tensor)
            probabilities = torch.softmax(logits, dim=1)
            
        return probabilities.numpy()


class DomainAdversarialNetwork(BaseTransferModel):
    """
    Domain Adversarial Neural Network (DANN) for Transfer Learning.
    
    This approach learns domain-invariant features through adversarial training
    between a feature extractor and domain classifier.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.params = config['models']['domain_adversarial']
        self.lambda_domain = self.params['lambda_domain']
        
        self.feature_extractor = None
        self.label_predictor = None
        self.domain_classifier = None
        
    def _build_feature_extractor(self, input_dim: int) -> nn.Module:
        """Build the feature extractor network."""
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64)
        )
    
    def _build_label_predictor(self) -> nn.Module:
        """Build the label predictor network."""
        return nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)
        )
    
    def _build_domain_classifier(self) -> nn.Module:
        """Build the domain classifier network."""
        return nn.Sequential(
            GradientReversalLayer(self.lambda_domain),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)  # Source vs Target domain
        )
    
    def fit(self, source_data: Tuple[np.ndarray, np.ndarray], 
            target_data: Tuple[np.ndarray, np.ndarray]) -> 'DomainAdversarialNetwork':
        """
        Fit the domain adversarial network.
        
        Args:
            source_data: Tuple of (X_source, y_source)
            target_data: Tuple of (X_target, y_target)
            
        Returns:
            Self for method chaining
        """
        X_source, y_source = source_data
        X_target, y_target = target_data
        
        # Standardize features
        X_source_scaled = self.scaler.fit_transform(X_source)
        X_target_scaled = self.scaler.transform(X_target)
        
        # Convert to tensors
        X_source_tensor = torch.FloatTensor(X_source_scaled)
        y_source_tensor = torch.LongTensor(y_source)
        X_target_tensor = torch.FloatTensor(X_target_scaled)
        y_target_tensor = torch.LongTensor(y_target)
        
        # Create domain labels (0 for source, 1 for target)
        domain_source = torch.zeros(len(X_source), dtype=torch.long)
        domain_target = torch.ones(len(X_target), dtype=torch.long)
        
        # Build networks
        input_dim = X_source.shape[1]
        self.feature_extractor = self._build_feature_extractor(input_dim)
        self.label_predictor = self._build_label_predictor()
        self.domain_classifier = self._build_domain_classifier()
        
        # Optimizers
        optimizer_fe = optim.Adam(self.feature_extractor.parameters(), 
                                 lr=self.params['learning_rate'])
        optimizer_lp = optim.Adam(self.label_predictor.parameters(), 
                                 lr=self.params['learning_rate'])
        optimizer_dc = optim.Adam(self.domain_classifier.parameters(), 
                                 lr=self.params['learning_rate'])
        
        label_criterion = nn.CrossEntropyLoss()
        domain_criterion = nn.CrossEntropyLoss()
        
        for epoch in range(self.params['n_epochs']):
            # Forward pass for source domain
            source_features = self.feature_extractor(X_source_tensor)
            source_label_pred = self.label_predictor(source_features)
            source_domain_pred = self.domain_classifier(source_features)
            
            # Forward pass for target domain
            target_features = self.feature_extractor(X_target_tensor)
            target_domain_pred = self.domain_classifier(target_features)
            
            # Label prediction loss (only on source domain)
            label_loss = label_criterion(source_label_pred, y_source_tensor)
            
            # Domain classification loss (both domains)
            domain_loss_source = domain_criterion(source_domain_pred, domain_source)
            domain_loss_target = domain_criterion(target_domain_pred, domain_target)
            domain_loss = domain_loss_source + domain_loss_target
            
            # Total loss
            total_loss = label_loss + domain_loss
            
            # Backward pass
            optimizer_fe.zero_grad()
            optimizer_lp.zero_grad()
            optimizer_dc.zero_grad()
            
            total_loss.backward()
            
            optimizer_fe.step()
            optimizer_lp.step()
            optimizer_dc.step()
            
            if epoch % 20 == 0:
                logger.debug(f"DANN Epoch {epoch}, Label Loss: {label_loss.item():.4f}, "
                           f"Domain Loss: {domain_loss.item():.4f}")
        
        self.is_fitted = True
        logger.info("Domain Adversarial Network model fitted successfully")
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using domain adversarial network.
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        with torch.no_grad():
            features = self.feature_extractor(X_tensor)
            logits = self.label_predictor(features)
            probabilities = torch.softmax(logits, dim=1)
            
        return probabilities.numpy()


class CausalTransferLearning(BaseTransferModel):
    """
    Causal Transfer Learning using Propensity Score Weighting.
    
    This approach uses causal inference principles to identify stable
    causal relationships that transfer across domains.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.params = config['models']['causal_transfer']
        
        self.propensity_model = None
        self.outcome_model = None
        self.propensity_scores = None
        
    def _estimate_propensity_scores(self, X_source: np.ndarray, X_target: np.ndarray) -> np.ndarray:
        """
        Estimate propensity scores for domain membership.
        
        Args:
            X_source: Source domain features
            X_target: Target domain features
            
        Returns:
            Propensity scores for all samples
        """
        # Combine source and target data
        X_combined = np.vstack([X_source, X_target])
        
        # Create domain labels (0 for source, 1 for target)
        domain_labels = np.hstack([
            np.zeros(len(X_source)),
            np.ones(len(X_target))
        ])
        
        # Fit propensity score model
        self.propensity_model = LogisticRegression(
            C=self.params['propensity_C'],
            max_iter=self.params['max_iter'],
            random_state=self.config['random_seeds']['model_init']
        )
        
        self.propensity_model.fit(X_combined, domain_labels)
        
        # Get propensity scores
        propensity_scores = self.propensity_model.predict_proba(X_combined)[:, 1]
        
        return propensity_scores
    
    def _compute_weights(self, propensity_scores: np.ndarray, 
                        domain_labels: np.ndarray) -> np.ndarray:
        """
        Compute inverse propensity weights.
        
        Args:
            propensity_scores: Estimated propensity scores
            domain_labels: Domain membership labels
            
        Returns:
            Sample weights
        """
        weights = np.zeros_like(propensity_scores)
        
        # Source domain weights: 1 / (1 - e(x))
        source_mask = domain_labels == 0
        weights[source_mask] = 1.0 / (1.0 - propensity_scores[source_mask] + 1e-8)
        
        # Target domain weights: 1 / e(x)
        target_mask = domain_labels == 1
        weights[target_mask] = 1.0 / (propensity_scores[target_mask] + 1e-8)
        
        # Normalize weights
        weights = weights / np.mean(weights)
        
        # Clip extreme weights
        weights = np.clip(weights, 0.1, 10.0)
        
        return weights
    
    def fit(self, source_data: Tuple[np.ndarray, np.ndarray], 
            target_data: Tuple[np.ndarray, np.ndarray]) -> 'CausalTransferLearning':
        """
        Fit the causal transfer learning model.
        
        Args:
            source_data: Tuple of (X_source, y_source)
            target_data: Tuple of (X_target, y_target)
            
        Returns:
            Self for method chaining
        """
        X_source, y_source = source_data
        X_target, y_target = target_data
        
        # Standardize features
        X_source_scaled = self.scaler.fit_transform(X_source)
        X_target_scaled = self.scaler.transform(X_target)
        
        # Estimate propensity scores
        propensity_scores = self._estimate_propensity_scores(X_source_scaled, X_target_scaled)
        
        # Create domain labels
        domain_labels = np.hstack([
            np.zeros(len(X_source)),
            np.ones(len(X_target))
        ])
        
        # Compute weights
        weights = self._compute_weights(propensity_scores, domain_labels)
        
        # Store propensity scores for source domain
        self.propensity_scores = propensity_scores[:len(X_source)]
        
        # Fit weighted outcome model on source domain
        source_weights = weights[:len(X_source)]
        
        self.outcome_model = LogisticRegression(
            C=self.params['outcome_C'],
            max_iter=self.params['max_iter'],
            random_state=self.config['random_seeds']['model_init']
        )
        
        self.outcome_model.fit(X_source_scaled, y_source, sample_weight=source_weights)
        
        self.is_fitted = True
        logger.info("Causal Transfer Learning model fitted successfully")
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using causal transfer learning.
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.outcome_model.predict_proba(X_scaled)


class TabTransformer(BaseTransferModel):
    """
    TabTransformer: Transformer Architecture for Tabular Data.
    
    This approach uses self-attention mechanisms to capture complex
    feature interactions in tabular healthcare data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.params = config['models']['tab_transformer']
        
        self.transformer_model = None
        self.categorical_features = []
        self.numerical_features = []
        
    def _build_transformer_model(self, input_dim: int) -> nn.Module:
        """Build the TabTransformer model."""
        
        class MultiHeadAttention(nn.Module):
            def __init__(self, d_model, n_heads):
                super().__init__()
                self.d_model = d_model
                self.n_heads = n_heads
                self.d_k = d_model // n_heads
                
                self.W_q = nn.Linear(d_model, d_model)
                self.W_k = nn.Linear(d_model, d_model)
                self.W_v = nn.Linear(d_model, d_model)
                self.W_o = nn.Linear(d_model, d_model)
                
            def forward(self, x):
                batch_size, seq_len, d_model = x.size()
                
                Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
                K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
                V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
                
                scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
                attention_weights = torch.softmax(scores, dim=-1)
                
                context = torch.matmul(attention_weights, V)
                context = context.transpose(1, 2).contiguous().view(
                    batch_size, seq_len, d_model
                )
                
                output = self.W_o(context)
                return output
        
        class TransformerBlock(nn.Module):
            def __init__(self, d_model, n_heads, d_ff, dropout):
                super().__init__()
                self.attention = MultiHeadAttention(d_model, n_heads)
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)
                
                self.feed_forward = nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model)
                )
                
                self.dropout = nn.Dropout(dropout)
                
            def forward(self, x):
                # Self-attention with residual connection
                attn_output = self.attention(x)
                x = self.norm1(x + self.dropout(attn_output))
                
                # Feed-forward with residual connection
                ff_output = self.feed_forward(x)
                x = self.norm2(x + self.dropout(ff_output))
                
                return x
        
        class TabTransformerModel(nn.Module):
            def __init__(self, input_dim, d_model, n_heads, n_layers, d_ff, dropout):
                super().__init__()
                self.input_projection = nn.Linear(input_dim, d_model)
                
                self.transformer_blocks = nn.ModuleList([
                    TransformerBlock(d_model, n_heads, d_ff, dropout)
                    for _ in range(n_layers)
                ])
                
                self.classifier = nn.Sequential(
                    nn.Linear(d_model, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, 2)
                )
                
            def forward(self, x):
                # Project input to model dimension
                x = self.input_projection(x)
                
                # Add sequence dimension for transformer
                x = x.unsqueeze(1)  # (batch_size, 1, d_model)
                
                # Apply transformer blocks
                for transformer_block in self.transformer_blocks:
                    x = transformer_block(x)
                
                # Remove sequence dimension and classify
                x = x.squeeze(1)  # (batch_size, d_model)
                output = self.classifier(x)
                
                return output
        
        return TabTransformerModel(
            input_dim=input_dim,
            d_model=self.params['d_model'],
            n_heads=self.params['n_heads'],
            n_layers=self.params['n_layers'],
            d_ff=self.params['d_ff'],
            dropout=self.params['dropout']
        )
    
    def fit(self, source_data: Tuple[np.ndarray, np.ndarray], 
            target_data: Tuple[np.ndarray, np.ndarray]) -> 'TabTransformer':
        """
        Fit the TabTransformer model.
        
        Args:
            source_data: Tuple of (X_source, y_source)
            target_data: Tuple of (X_target, y_target)
            
        Returns:
            Self for method chaining
        """
        X_source, y_source = source_data
        X_target, y_target = target_data
        
        # Standardize features
        X_source_scaled = self.scaler.fit_transform(X_source)
        X_target_scaled = self.scaler.transform(X_target)
        
        # Combine source and target data for training
        X_combined = np.vstack([X_source_scaled, X_target_scaled])
        y_combined = np.hstack([y_source, y_target])
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_combined)
        y_tensor = torch.LongTensor(y_combined)
        
        # Build model
        input_dim = X_source.shape[1]
        self.transformer_model = self._build_transformer_model(input_dim)
        
        # Training
        optimizer = optim.Adam(self.transformer_model.parameters(), 
                             lr=self.params['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(self.params['n_epochs']):
            # Forward pass
            logits = self.transformer_model(X_tensor)
            loss = criterion(logits, y_tensor)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                logger.debug(f"TabTransformer Epoch {epoch}, Loss: {loss.item():.4f}")
        
        self.is_fitted = True
        logger.info("TabTransformer model fitted successfully")
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using TabTransformer.
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        with torch.no_grad():
            logits = self.transformer_model(X_tensor)
            probabilities = torch.softmax(logits, dim=1)
            
        return probabilities.numpy()


class MetaEnsemble(BaseTransferModel):
    """
    Meta-Ensemble: Ensemble of Multiple Transfer Learning Approaches.
    
    This approach combines multiple base transfer learning models using
    meta-learning to optimize ensemble weights for the target domain.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.params = config['models']['meta_ensemble']
        
        self.base_models = {}
        self.meta_learner = None
        self.ensemble_weights = None
        
    def _initialize_base_models(self) -> Dict[str, BaseTransferModel]:
        """Initialize base transfer learning models."""
        base_models = {}
        
        # Source-only transfer
        base_models['source_only'] = SourceOnlyTransfer(self.config)
        
        # Logistic regression with different regularization
        base_models['logistic_l1'] = SourceOnlyTransfer({
            **self.config,
            'models': {
                'source_only': {
                    'C': 0.1,
                    'max_iter': 1000,
                    'solver': 'liblinear',
                    'penalty': 'l1'
                }
            }
        })
        
        # Random Forest
        class RandomForestTransfer(BaseTransferModel):
            def __init__(self, config):
                super().__init__(config)
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=config['random_seeds']['model_init']
                )
            
            def fit(self, source_data, target_data):
                X_source, y_source = source_data
                X_source_scaled = self.scaler.fit_transform(X_source)
                self.model.fit(X_source_scaled, y_source)
                self.is_fitted = True
                return self
            
            def predict_proba(self, X):
                X_scaled = self.scaler.transform(X)
                return self.model.predict_proba(X_scaled)
        
        base_models['random_forest'] = RandomForestTransfer(self.config)
        
        # Gradient Boosting
        class GradientBoostingTransfer(BaseTransferModel):
            def __init__(self, config):
                super().__init__(config)
                self.model = GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=config['random_seeds']['model_init']
                )
            
            def fit(self, source_data, target_data):
                X_source, y_source = source_data
                X_source_scaled = self.scaler.fit_transform(X_source)
                self.model.fit(X_source_scaled, y_source)
                self.is_fitted = True
                return self
            
            def predict_proba(self, X):
                X_scaled = self.scaler.transform(X)
                return self.model.predict_proba(X_scaled)
        
        base_models['gradient_boosting'] = GradientBoostingTransfer(self.config)
        
        return base_models
    
    def _train_meta_learner(self, meta_features: np.ndarray, 
                           meta_targets: np.ndarray) -> None:
        """Train the meta-learner to combine base model predictions."""
        self.meta_learner = LogisticRegression(
            C=self.params['meta_C'],
            max_iter=self.params['max_iter'],
            random_state=self.config['random_seeds']['model_init']
        )
        
        self.meta_learner.fit(meta_features, meta_targets)
    
    def fit(self, source_data: Tuple[np.ndarray, np.ndarray], 
            target_data: Tuple[np.ndarray, np.ndarray]) -> 'MetaEnsemble':
        """
        Fit the meta-ensemble model.
        
        Args:
            source_data: Tuple of (X_source, y_source)
            target_data: Tuple of (X_target, y_target)
            
        Returns:
            Self for method chaining
        """
        X_source, y_source = source_data
        X_target, y_target = target_data
        
        # Standardize features
        X_source_scaled = self.scaler.fit_transform(X_source)
        X_target_scaled = self.scaler.transform(X_target)
        
        # Initialize and train base models
        self.base_models = self._initialize_base_models()
        
        logger.info("Training base models for meta-ensemble")
        for name, model in self.base_models.items():
            logger.debug(f"Training base model: {name}")
            model.fit(source_data, target_data)
        
        # Generate meta-features using cross-validation on target domain
        cv = StratifiedKFold(n_splits=self.params['cv_folds'], shuffle=True, 
                           random_state=self.config['random_seeds']['cv'])
        
        meta_features = []
        meta_targets = []
        
        for train_idx, val_idx in cv.split(X_target_scaled, y_target):
            X_train_fold = X_target_scaled[train_idx]
            y_train_fold = y_target[train_idx]
            X_val_fold = X_target_scaled[val_idx]
            y_val_fold = y_target[val_idx]
            
            # Train base models on fold training data
            fold_predictions = []
            for name, model in self.base_models.items():
                # Create a copy of the model for this fold
                fold_model = type(model)(self.config)
                fold_model.fit((X_source_scaled, y_source), (X_train_fold, y_train_fold))
                
                # Get predictions on validation fold
                fold_pred = fold_model.predict_proba(X_val_fold)[:, 1]
                fold_predictions.append(fold_pred)
            
            # Stack predictions as meta-features
            fold_meta_features = np.column_stack(fold_predictions)
            meta_features.append(fold_meta_features)
            meta_targets.append(y_val_fold)
        
        # Combine all meta-features and targets
        meta_features = np.vstack(meta_features)
        meta_targets = np.hstack(meta_targets)
        
        # Train meta-learner
        self._train_meta_learner(meta_features, meta_targets)
        
        self.is_fitted = True
        logger.info("Meta-Ensemble model fitted successfully")
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using meta-ensemble.
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get predictions from all base models
        base_predictions = []
        for name, model in self.base_models.items():
            pred = model.predict_proba(X)[:, 1]
            base_predictions.append(pred)
        
        # Stack as meta-features
        meta_features = np.column_stack(base_predictions)
        
        # Get final predictions from meta-learner
        final_predictions = self.meta_learner.predict_proba(meta_features)
        
        return final_predictions


class TransferLearningPipeline:
    """
    Main pipeline for training and managing all transfer learning models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the transfer learning pipeline.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.models = {}
        
        # Initialize all model types
        self.model_classes = {
            'source_only': SourceOnlyTransfer,
            'prototypical_networks': PrototypicalNetworks,
            'maml': MAML,
            'domain_adversarial': DomainAdversarialNetwork,
            'causal_transfer': CausalTransferLearning,
            'tab_transformer': TabTransformer,
            'meta_ensemble': MetaEnsemble,
        }
    
    def train_all_models(self, source_data: Tuple[np.ndarray, np.ndarray], 
                        target_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, BaseTransferModel]:
        """
        Train all transfer learning models.
        
        Args:
            source_data: Tuple of (X_source, y_source)
            target_data: Tuple of (X_target, y_target)
            
        Returns:
            Dictionary of trained models
        """
        logger.info("Training all transfer learning models")
        
        for model_name, model_class in self.model_classes.items():
            logger.info(f"Training {model_name}")
            
            try:
                model = model_class(self.config)
                model.fit(source_data, target_data)
                self.models[model_name] = model
                logger.info(f"Successfully trained {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {str(e)}")
                continue
        
        logger.info(f"Successfully trained {len(self.models)} models")
        return self.models
    
    def get_model(self, model_name: str) -> Optional[BaseTransferModel]:
        """
        Get a specific trained model.
        
        Args:
            model_name: Name of the model to retrieve
            
        Returns:
            Trained model or None if not found
        """
        return self.models.get(model_name)
    
    def get_all_models(self) -> Dict[str, BaseTransferModel]:
        """
        Get all trained models.
        
        Returns:
            Dictionary of all trained models
        """
        return self.models.copy()

