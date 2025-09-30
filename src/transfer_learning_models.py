"""Transfer learning model implementations for Medicaid risk prediction."""

from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data_preprocessing import DomainDataBundle

logger = logging.getLogger(__name__)


def set_global_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - depends on environment
        torch.cuda.manual_seed_all(seed)


class BaseTransferModel(ABC):
    """Abstract base class for transfer learning approaches."""

    def __init__(self, config: Dict[str, Any], model_key: str):
        self.config = config
        self.model_key = model_key
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.random_seed = config.get("random_seeds", {}).get("model_init", 42)
        set_global_seeds(self.random_seed)

    @abstractmethod
    def fit(self, source_data: DomainDataBundle, target_data: DomainDataBundle) -> "BaseTransferModel":
        """Train the model using source and target domain data."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities for the positive class."""

    def to(self, device: torch.device) -> "BaseTransferModel":  # pragma: no cover - interface helper
        return self


class SourceOnlyTransfer(BaseTransferModel):
    """Train on source data only and evaluate on target domain."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "source_only")
        params = config.get("models", {}).get("source_only", {})
        self.model = LogisticRegression(
            C=params.get("C", 1.0),
            max_iter=params.get("max_iter", 1000),
            solver=params.get("solver", "lbfgs"),
            class_weight=params.get("class_weight"),
        )

    def fit(self, source_data: DomainDataBundle, target_data: DomainDataBundle) -> "SourceOnlyTransfer":
        X_train = self.scaler.fit_transform(source_data.X_train)
        self.model.fit(X_train, source_data.y_train)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class TargetOnlyTransfer(BaseTransferModel):
    """Train a target-domain model using adaptation data only."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "target_only")
        params = config.get("models", {}).get("target_only", {})
        self.model = LogisticRegression(
            C=params.get("C", 1.0),
            max_iter=params.get("max_iter", 1000),
            solver=params.get("solver", "lbfgs"),
            class_weight=params.get("class_weight"),
        )

    def fit(self, source_data: DomainDataBundle, target_data: DomainDataBundle) -> "TargetOnlyTransfer":
        if target_data.support_X is None or len(target_data.support_X) < 10:
            logger.warning("Adaptation set too small; falling back to full target training set.")
            X_train = target_data.X_train
            y_train = target_data.y_train
        else:
            X_train = target_data.support_X
            y_train = target_data.support_y
        X_train = self.scaler.fit_transform(X_train)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class PrototypicalNetworks(BaseTransferModel):
    """Few-shot adaptation via prototypical representations."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "prototypical_networks")
        params = config.get("models", {}).get("prototypical_networks", {})
        self.embedding_dim = params.get("embedding_dim", 64)
        self.n_epochs = params.get("n_epochs", 50)
        self.learning_rate = params.get("learning_rate", 1e-3)
        self.n_support = params.get("n_support", 10)
        self.embedding = None
        self.prototypes: Dict[str, torch.Tensor] = {}

    def _build_embedding(self, input_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.embedding_dim),
        )

    @staticmethod
    def _compute_prototypes(embeddings: torch.Tensor, labels: torch.Tensor) -> Dict[int, torch.Tensor]:
        prototypes: Dict[int, torch.Tensor] = {}
        for label in labels.unique():
            mask = labels == label
            if mask.sum() == 0:
                continue
            prototypes[int(label.item())] = embeddings[mask].mean(dim=0)
        return prototypes

    @staticmethod
    def _euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.cdist(x, y.unsqueeze(0)).squeeze(1)

    def fit(self, source_data: DomainDataBundle, target_data: DomainDataBundle) -> "PrototypicalNetworks":
        X_source = torch.FloatTensor(source_data.X_train)
        y_source = torch.LongTensor(source_data.y_train)
        X_target = torch.FloatTensor(target_data.X_train)
        y_target = torch.LongTensor(target_data.y_train)

        self.embedding = self._build_embedding(X_source.shape[1])
        optimizer = optim.Adam(self.embedding.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            source_embed = self.embedding(X_source)
            target_embed = self.embedding(X_target)

            source_proto = self._compute_prototypes(source_embed, y_source)
            n_support = min(self.n_support, len(X_target) // 2)
            indices = torch.randperm(len(X_target))
            support_idx = indices[:n_support]
            query_idx = indices[n_support:]
            support_embed = target_embed[support_idx]
            query_embed = target_embed[query_idx]
            support_labels = y_target[support_idx]
            query_labels = y_target[query_idx]
            target_proto = self._compute_prototypes(support_embed, support_labels)

            logits = []
            for class_label in [0, 1]:
                prototype = target_proto.get(class_label, source_proto.get(class_label))
                if prototype is None:
                    logits.append(torch.full((len(query_embed),), -1.0))
                    continue
                dist = self._euclidean_distance(query_embed, prototype)
                logits.append(-dist)
            logits = torch.stack(logits, dim=1)
            loss = criterion(logits, query_labels)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            self.prototypes["source"] = self._compute_prototypes(self.embedding(X_source), y_source)
            self.prototypes["target"] = self._compute_prototypes(self.embedding(X_target), y_target)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            embed = self.embedding(X_tensor)
            proto = self.prototypes.get("target", self.prototypes.get("source"))
            logits = []
            for class_label in [0, 1]:
                centroid = proto.get(class_label)
                if centroid is None:
                    logits.append(torch.full((len(embed),), -1.0))
                    continue
                dist = self._euclidean_distance(embed, centroid)
                logits.append(-dist)
            logits = torch.stack(logits, dim=1)
            probs = F.softmax(logits, dim=1)
        return probs.numpy()


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float) -> torch.Tensor:
    return GradientReversal.apply(x, lambda_)


class DomainAdversarialNetwork(BaseTransferModel):
    """Gradient reversal network for domain-invariant feature learning."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "domain_adversarial")
        params = config.get("models", {}).get("domain_adversarial", {})
        self.lambda_domain = params.get("lambda_domain", 0.1)
        self.n_epochs = params.get("n_epochs", 50)
        self.learning_rate = params.get("learning_rate", 1e-3)
        self.feature_extractor: Optional[nn.Module] = None
        self.label_classifier: Optional[nn.Module] = None
        self.domain_classifier: Optional[nn.Module] = None

    def _build_networks(self, input_dim: int) -> None:
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.label_classifier = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
        self.domain_classifier = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))

    def fit(self, source_data: DomainDataBundle, target_data: DomainDataBundle) -> "DomainAdversarialNetwork":
        X_source = torch.FloatTensor(source_data.X_train)
        y_source = torch.LongTensor(source_data.y_train)
        X_target_full = torch.FloatTensor(target_data.X_train)
        X_target_support = torch.FloatTensor(target_data.support_X)
        y_target_support = torch.LongTensor(target_data.support_y)

        self._build_networks(X_source.shape[1])
        params = list(self.feature_extractor.parameters()) + list(self.label_classifier.parameters())
        optimizer = optim.Adam(params, lr=self.learning_rate)
        domain_optimizer = optim.Adam(self.domain_classifier.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.n_epochs):
            self.feature_extractor.train()
            self.label_classifier.train()
            self.domain_classifier.train()

            optimizer.zero_grad()
            domain_optimizer.zero_grad()

            features_source = self.feature_extractor(X_source)
            features_support = self.feature_extractor(X_target_support)
            label_logits_source = self.label_classifier(features_source)
            label_logits_support = self.label_classifier(features_support)
            label_loss = criterion(label_logits_source, y_source) + criterion(label_logits_support, y_target_support)

            features_target = self.feature_extractor(X_target_full)
            combined_features = torch.cat([features_source, features_target], dim=0)
            reversed_features = grad_reverse(combined_features, self.lambda_domain)
            domain_logits = self.domain_classifier(reversed_features)
            domain_labels = torch.cat(
                [torch.zeros(len(X_source), dtype=torch.long), torch.ones(len(X_target_full), dtype=torch.long)],
                dim=0,
            )
            domain_loss = criterion(domain_logits, domain_labels)

            total_loss = label_loss + domain_loss
            total_loss.backward()
            optimizer.step()
            domain_optimizer.step()

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            features = self.feature_extractor(X_tensor)
            logits = self.label_classifier(features)
            probabilities = F.softmax(logits, dim=1)
        return probabilities.numpy()


class CausalTransferLearning(BaseTransferModel):
    """Reweight source domain using propensity scores to mimic target population."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "causal_transfer")
        params = config.get("models", {}).get("causal_transfer", {})
        self.propensity_model = LogisticRegression(max_iter=1000)
        self.outcome_model = LogisticRegression(max_iter=1000)
        self.backdoor_adjustment = params.get("backdoor_adjustment", True)

    def fit(self, source_data: DomainDataBundle, target_data: DomainDataBundle) -> "CausalTransferLearning":
        X_source = source_data.X_train
        y_source = source_data.y_train
        X_target = target_data.X_train

        combined_X = np.vstack([X_source, X_target])
        domain_labels = np.concatenate([np.zeros(len(X_source)), np.ones(len(X_target))])
        self.propensity_model.fit(combined_X, domain_labels)
        propensity_scores = self.propensity_model.predict_proba(X_source)[:, 1]
        weights = propensity_scores / np.clip(1 - propensity_scores, 1e-3, 1)
        weights = np.clip(weights, 0, 50)

        if target_data.support_X is not None:
            X_train = np.vstack([X_source, target_data.support_X])
            y_train = np.concatenate([y_source, target_data.support_y])
            sample_weight = np.concatenate([weights, np.full(len(target_data.support_y), 1.0)])
        else:
            X_train = X_source
            y_train = y_source
            sample_weight = weights

        self.outcome_model.fit(self.scaler.fit_transform(X_train), y_train, sample_weight=sample_weight)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X_scaled = self.scaler.transform(X)
        return self.outcome_model.predict_proba(X_scaled)


class TabTransformer(BaseTransferModel):
    """Feed-forward transformer-style encoder for tabular data."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "tabtransformer")
        params = config.get("models", {}).get("tabtransformer", {})
        self.learning_rate = params.get("learning_rate", 1e-3)
        self.n_epochs = params.get("n_epochs", 50)
        self.dropout = params.get("dropout", 0.1)
        self.hidden_dim = params.get("hidden_dim", 128)
        self.model: Optional[nn.Module] = None

    def _build_model(self, input_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 2),
        )

    def fit(self, source_data: DomainDataBundle, target_data: DomainDataBundle) -> "TabTransformer":
        if target_data.support_X is not None and len(target_data.support_X) > 0:
            combined_X = np.vstack([source_data.X_train, target_data.support_X])
            combined_y = np.concatenate([source_data.y_train, target_data.support_y])
        else:
            combined_X = source_data.X_train
            combined_y = source_data.y_train
        X_train = torch.FloatTensor(combined_X)
        y_train = torch.LongTensor(combined_y)
        self.model = self._build_model(X_train.shape[1])
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            logits = self.model(X_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = F.softmax(logits, dim=1)
        return probs.numpy()


class EnhancedMAML(BaseTransferModel):
    """Enhanced MAML with optional regularisation components."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "enhanced_maml")
        params = config.get("models", {}).get("enhanced_maml", config.get("models", {}).get("maml", {}))
        self.inner_lr = params.get("inner_lr", 0.01)
        self.outer_lr = params.get("outer_lr", 1e-3)
        self.n_inner_steps = params.get("n_inner_steps", 5)
        self.n_epochs = params.get("n_epochs", 50)
        self.first_order = params.get("first_order", True)
        self.enable_meta_learning = params.get("meta_learning_adaptation", True)
        self.enable_feature_alignment = params.get("feature_alignment", True)
        self.enable_temporal_stability = params.get("temporal_stability", True)
        self.meta_model: Optional[nn.Module] = None
        self.feature_extractor: Optional[nn.Module] = None
        self.support_cache: Optional[torch.Tensor] = None
        self.support_labels: Optional[torch.Tensor] = None

    def _build_model(self, input_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def _inner_loop(self, model: nn.Module, support_x: torch.Tensor, support_y: torch.Tensor) -> nn.Module:
        adapted = copy.deepcopy(model)
        optimizer = optim.SGD(adapted.parameters(), lr=self.inner_lr)
        criterion = nn.CrossEntropyLoss()
        for _ in range(self.n_inner_steps):
            optimizer.zero_grad()
            logits = adapted(support_x)
            loss = criterion(logits, support_y)
            loss.backward()
            optimizer.step()
        return adapted

    def _feature_alignment_penalty(self, features: torch.Tensor) -> torch.Tensor:
        if features.size(0) < 2:
            return torch.tensor(0.0)
        covariance = torch.cov(features.T)
        off_diag = covariance - torch.diag(torch.diag(covariance))
        return off_diag.abs().mean()

    def _temporal_stability_penalty(self, logits: torch.Tensor) -> torch.Tensor:
        diffs = logits[1:] - logits[:-1]
        return torch.mean(diffs.pow(2)) if diffs.numel() > 0 else torch.tensor(0.0)

    def fit(self, source_data: DomainDataBundle, target_data: DomainDataBundle) -> "EnhancedMAML":
        X_source = torch.FloatTensor(source_data.X_train)
        y_source = torch.LongTensor(source_data.y_train)
        support_x = torch.FloatTensor(target_data.support_X)
        support_y = torch.LongTensor(target_data.support_y)
        query_x = torch.FloatTensor(target_data.query_X)
        query_y = torch.LongTensor(target_data.query_y)

        self.meta_model = self._build_model(X_source.shape[1])
        if isinstance(self.meta_model, nn.Sequential):
            self.feature_extractor = self.meta_model[:-1]
        else:
            self.feature_extractor = None
        meta_optimizer = optim.Adam(self.meta_model.parameters(), lr=self.outer_lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.n_epochs):
            meta_optimizer.zero_grad()
            logits_source = self.meta_model(X_source)
            loss_source = criterion(logits_source, y_source)

            adapted_model = self._inner_loop(self.meta_model, support_x, support_y) if self.enable_meta_learning else self.meta_model
            query_logits = adapted_model(query_x)
            loss_query = criterion(query_logits, query_y)

            total_loss = 0.5 * loss_source + loss_query
            if self.enable_feature_alignment and self.feature_extractor is not None:
                features = self.feature_extractor(torch.cat([support_x, query_x], dim=0))
                total_loss += 0.01 * self._feature_alignment_penalty(features)
            if self.enable_temporal_stability:
                total_loss += 0.01 * self._temporal_stability_penalty(query_logits)

            total_loss.backward()
            meta_optimizer.step()

        self.support_cache = support_x
        self.support_labels = support_y
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X_tensor = torch.FloatTensor(X)
        adapted_model = self._inner_loop(self.meta_model, self.support_cache, self.support_labels) if self.enable_meta_learning else self.meta_model
        with torch.no_grad():
            logits = adapted_model(X_tensor)
            probs = F.softmax(logits, dim=1)
        return probs.numpy()


class MetaEnsemble(BaseTransferModel):
    """Stacks base learner predictions using a logistic meta-learner."""

    def __init__(self, config: Dict[str, Any], base_models: Dict[str, BaseTransferModel]):
        super().__init__(config, "meta_ensemble")
        self.base_models = base_models
        params = config.get("models", {}).get("meta_ensemble", {})
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=params.get("n_epochs", 200))

    def fit(self, source_data: DomainDataBundle, target_data: DomainDataBundle) -> "MetaEnsemble":
        if target_data.query_X is None:
            raise ValueError("Target domain query set required for meta-ensemble training")
        X_meta = []
        for name, model in self.base_models.items():
            if name == "meta_ensemble":
                continue
            probs = model.predict_proba(target_data.query_X)[:, 1]
            X_meta.append(probs)
        X_meta = np.vstack(X_meta).T
        X_meta_scaled = self.scaler.fit_transform(X_meta)
        self.model.fit(X_meta_scaled, target_data.query_y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        stacked = []
        for name, model in self.base_models.items():
            if name == "meta_ensemble":
                continue
            probs = model.predict_proba(X)[:, 1]
            stacked.append(probs)
        X_meta = np.vstack(stacked).T
        X_meta_scaled = self.scaler.transform(X_meta)
        probs = self.model.predict_proba(X_meta_scaled)
        return probs


class TransferLearningPipeline:
    """Coordinates training of all transfer learning approaches."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models: Dict[str, BaseTransferModel] = {}

    def train_all_models(
        self, source_data: DomainDataBundle, target_data: DomainDataBundle
    ) -> Dict[str, BaseTransferModel]:
        model_registry: Dict[str, Any] = {
            "source_only": SourceOnlyTransfer,
            "target_only": TargetOnlyTransfer,
            "prototypical_networks": PrototypicalNetworks,
            "domain_adversarial": DomainAdversarialNetwork,
            "causal_transfer": CausalTransferLearning,
            "tabtransformer": TabTransformer,
            "enhanced_maml": EnhancedMAML,
        }

        for name, constructor in model_registry.items():
            try:
                logger.info("Training %s", name)
                model = constructor(self.config)
                model.fit(source_data, target_data)
                self.models[name] = model
            except Exception as exc:  # pragma: no cover - keep training resilient
                logger.error("%s training failed: %s", name, exc)

        if self.models:
            try:
                logger.info("Training meta ensemble")
                ensemble = MetaEnsemble(self.config, self.models)
                ensemble.fit(source_data, target_data)
                self.models["meta_ensemble"] = ensemble
            except Exception as exc:
                logger.error("Meta ensemble training failed: %s", exc)

        return self.models

    def get_model(self, model_name: str) -> Optional[BaseTransferModel]:
        return self.models.get(model_name)

    def get_all_models(self) -> Dict[str, BaseTransferModel]:
        return dict(self.models)
