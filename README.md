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

## 🎯 Key Results

### Primary Findings
- **Enhanced Meta-Learning MAML** achieved the best performance
- **194.4% improvement** in Youden's J Index over naive transfer
- **Consistent gains** across all performance metrics
- **Significant improvements** in clinical utility metrics

### Performance Metrics (Best Model)
- **AUC-ROC**: 0.523 (95% CI: 0.498-0.548)
- **Youden's J**: 0.052 (95% CI: 0.041-0.063)
- **F1-Score**: 0.541 (95% CI: 0.518-0.564)
- **Number Needed to Treat**: 22.7 (95% CI: 19.8-25.6)

### Feature Importance
1. **Clinical Features**: 25.9% importance (Charlson score, mental health)
2. **Healthcare Utilization**: 22.7% importance (prior ED visits, hospitalizations)
3. **Meta-Learning Adaptation**: 17.9% importance (domain-specific adjustments)

## 📈 Generated Outputs

### Tables
- **Table 1**: Demographic and clinical characteristics
- **Table 2**: Clinical utility and performance metrics
- **Supplementary Tables S1-S8**: Detailed analysis results

### Figures
- **Figure 1**: Clinical utility comparison
- **Figure 2**: Sensitivity-specificity analysis
- **Figure 3**: Multi-metric performance comparison
- **Supplementary Figures**: Calibration, ablation, feature importance

### Reports
- **Final Analysis Report**: Comprehensive markdown report
- **Visualization Summary**: Figure generation summary
- **Statistical Results**: Detailed statistical testing results

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

## 🤝 Contributing

We welcome contributions to improve the codebase and extend the analysis:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-analysis`
3. **Make changes and add tests**
4. **Submit a pull request**

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes
- Include unit tests for new functionality
- Update documentation as needed

## 📄 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{medicaid_transfer_learning_2024,
  title={Generalizing Healthcare Risk Prediction Models: A Prospective Evaluation of Transfer Learning for Predicting Acute Care Use in Medicaid Populations},
  author={[Authors]},
  journal={npj Health Systems},
  year={2024},
  doi={[DOI]}
}
```

## 📞 Support

### Getting Help
- **Issues**: Report bugs and request features via GitHub Issues
- **Discussions**: Ask questions in GitHub Discussions
- **Email**: Contact the corresponding author for research-related questions

### Common Issues
1. **Memory Errors**: Reduce batch size or number of bootstrap iterations
2. **CUDA Errors**: Set device to 'cpu' in configuration
3. **Missing Dependencies**: Ensure all packages in requirements.txt are installed

## 📋 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Funding**: [Grant information]
- **Data Sources**: Medicaid administrative claims data
- **Computing Resources**: [Computing infrastructure]
- **Collaborators**: [Institutional collaborations]

## 🔄 Version History

- **v1.0.0**: Initial release with complete analysis pipeline
- **v1.1.0**: Added calibration analysis and improved documentation
- **v1.2.0**: Enhanced visualization and statistical testing

---

**Note**: This repository contains synthetic data for demonstration purposes. For access to real Medicaid data, please follow appropriate data use agreements and IRB protocols.

