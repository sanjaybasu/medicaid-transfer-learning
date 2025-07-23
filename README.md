# Medicaid Transfer Learning for Healthcare Risk Prediction

This repository contains the complete implementation and analysis code for the research paper "Generalizing Healthcare Risk Prediction Models: A Prospective Evaluation of Transfer Learning for Predicting Acute Care Use in Medicaid Populations".

## 📋 Overview

This study evaluates seven transfer learning approaches for predicting acute care utilization in Medicaid populations across different states. The research addresses the critical challenge of model generalizability in healthcare prediction when deploying models across different healthcare systems and populations.

## 🏗️ Repository Structure

```
medicaid_transfer_learning_reproducible/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment specification
├── config/                           # Configuration files
│   ├── model_config.yaml            # Model hyperparameters and settings
│   └── data_config.yaml             # Data preprocessing configuration
├── src/                             # Source code
│   ├── __init__.py
│   ├── main_analysis.py             # Main analysis pipeline
│   ├── transfer_learning_models.py  # Transfer learning model implementations
│   ├── data_preprocessing.py        # Data preprocessing utilities
│   ├── evaluation_metrics.py        # Evaluation framework
│   ├── statistical_analysis.py      # Statistical testing and analysis
│   ├── calibration_analysis.py      # Model calibration assessment
│   ├── ablation_study.py           # Feature and component ablation
│   └── visualization.py            # Visualization generation
├── data/                           # Data directory (see data/README.md)
│   └── README.md                   # Data documentation
├── docs/                          # Documentation
│   └── replication_guide.md       # Detailed replication instructions
└── results/                       # Output directory (created during analysis)
    ├── figures/                   # Generated figures
    ├── tables/                    # Generated tables
    └── final_analysis_report.md   # Comprehensive analysis report
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ or Anaconda/Miniconda
- 8GB+ RAM recommended
- CUDA-compatible GPU (optional, for faster training)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/medicaid_transfer_learning_reproducible.git
cd medicaid_transfer_learning_reproducible
```

2. **Create and activate environment:**

**Option A: Using conda (recommended):**
```bash
conda env create -f environment.yml
conda activate medicaid_transfer_learning
```

**Option B: Using pip:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Analysis

**Complete analysis pipeline:**
```bash
python src/main_analysis.py
```

**With custom configurations:**
```bash
python src/main_analysis.py \
    --model-config config/model_config.yaml \
    --data-config config/data_config.yaml \
    --output-dir results \
    --log-level INFO
```

## 🔬 Transfer Learning Models

This repository implements seven state-of-the-art transfer learning approaches:

### 1. Source-Only Transfer (Baseline)
- **Description**: Direct application of source-trained model to target domain
- **Use Case**: Baseline comparison for transfer learning effectiveness
- **Implementation**: `SourceOnlyTransfer` class

### 2. Prototypical Networks
- **Description**: Few-shot learning with prototype-based classification
- **Key Features**: Embedding-based similarity learning, support/query split
- **Implementation**: `PrototypicalNetworks` class

### 3. Model-Agnostic Meta-Learning (MAML)
- **Description**: Gradient-based meta-learning for fast adaptation
- **Key Features**: Inner/outer loop optimization, first-order approximation
- **Implementation**: `MAML` class

### 4. Domain Adversarial Neural Networks
- **Description**: Adversarial training for domain-invariant features
- **Key Features**: Gradient reversal layer, domain classifier
- **Implementation**: `DomainAdversarialNetwork` class

### 5. Causal Transfer Learning
- **Description**: Propensity score-based domain adaptation
- **Key Features**: Covariate shift correction, causal inference
- **Implementation**: `CausalTransferLearning` class

### 6. TabTransformer
- **Description**: Transformer architecture for tabular data
- **Key Features**: Self-attention mechanism, categorical embeddings
- **Implementation**: `TabTransformer` class

### 7. Meta-Ensemble
- **Description**: Ensemble of base models with meta-learning
- **Key Features**: Stacking, cross-validation, multiple base learners
- **Implementation**: `MetaEnsemble` class

## 📊 Analysis Components

### Evaluation Framework
- **Clinical Metrics**: Youden's J Index, Number Needed to Treat (NNT)
- **Performance Metrics**: AUC-ROC, F1-Score, Matthews Correlation Coefficient
- **Statistical Testing**: Bootstrap confidence intervals, DeLong test
- **Multiple Comparisons**: Bonferroni correction

### Calibration Analysis
- **Metrics**: Expected Calibration Error (ECE), Brier Score
- **Visualization**: Reliability diagrams
- **Post-hoc Methods**: Platt scaling, Isotonic regression

### Ablation Studies
- **Feature Importance**: Permutation importance, drop-column analysis
- **Component Ablation**: Transfer learning component analysis
- **Feature Groups**: Demographic, clinical, utilization, temporal

### Statistical Analysis
- **Significance Testing**: Pairwise model comparisons
- **Effect Sizes**: Cohen's d, Cramér's V
- **Power Analysis**: Sample size calculations


## 🔧 Configuration

### Model Configuration (`config/model_config.yaml`)
```yaml
# Example configuration
models:
  maml:
    inner_lr: 0.01
    outer_lr: 0.001
    n_inner_steps: 5
    n_epochs: 100
    
evaluation:
  bootstrap:
    n_iterations: 1000
    confidence_level: 0.95
```

### Data Configuration (`config/data_config.yaml`)
```yaml
# Example configuration
data_generation:
  n_source_samples: 10000
  n_target_samples: 5000
  n_features: 127
  
preprocessing:
  missing_value_strategy: 'median'
  outlier_detection: 'iqr'
```

## 📚 Documentation

### Detailed Guides
- **[Replication Guide](docs/replication_guide.md)**: Step-by-step replication instructions
- **[Data Documentation](data/README.md)**: Data structure and preprocessing details
- **[API Documentation](docs/api_documentation.md)**: Code documentation and examples

### Research Compliance
- **TRIPOD-AI Guidelines**: Transparent reporting of AI prediction models
- **Reproducibility Standards**: Fixed random seeds, version control
- **Statistical Rigor**: Multiple comparison corrections, effect sizes


## 📄 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{medicaid_transfer_learning_2024,
  title={Generalizing Healthcare Risk Prediction Models: A Prospective Evaluation of Transfer Learning for Predicting Acute Care Use in Medicaid Populations},
  author={Sanjay Basu},
  year={2025}
}
```

## 📋 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 🔄 Version History

- **v1.0.0**: Initial release with complete analysis pipeline
- **v1.1.0**: Added calibration analysis and improved documentation
- **v1.2.0**: Enhanced visualization and statistical testing

