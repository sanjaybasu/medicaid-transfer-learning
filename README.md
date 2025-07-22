# Generalizing Healthcare Risk Prediction Models: Transfer Learning for Medicaid Populations

## Overview

This repository contains the complete implementation of transfer learning approaches for predicting acute care utilization in Medicaid populations across different states.

## Repository Structure

```
medicaid_transfer_learning_reproducible/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment specification
├── config/
│   ├── model_config.yaml             # Model hyperparameters
│   └── data_config.yaml              # Data processing configuration
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py          # Data loading and preprocessing
│   ├── transfer_learning_models.py   # All transfer learning implementations
│   ├── evaluation_metrics.py         # Performance evaluation functions
│   ├── statistical_analysis.py       # Bootstrap and significance testing
│   ├── calibration_analysis.py       # Model calibration methods
│   ├── ablation_study.py             # Feature and component ablation
│   ├── visualization.py              # Figure generation functions
│   └── main_analysis.py              # Main analysis pipeline
├── data/
│   ├── README.md                     # Data requirements and format
│   └── sample_data_format.csv       # Example data structure
├── results/
│   ├── figures/                      # Generated figures
│   ├── tables/                       # Generated tables
│   └── models/                       # Saved model artifacts
├── docs/
│   ├── methodology.md                # Detailed methodology
│   ├── replication_guide.md          # Step-by-step replication
│   └── extension_guide.md            # Guide for extending the work
└── tests/
    ├── test_models.py                # Unit tests for models
    ├── test_evaluation.py            # Unit tests for evaluation
    └── test_data_processing.py       # Unit tests for data processing
```

## Quick Start

### 1. Environment Setup

```bash
# Using conda (recommended)
conda env create -f environment.yml
conda activate medicaid_transfer_learning

# Or using pip
pip install -r requirements.txt
```

### 2. Data Preparation

Place your Medicaid claims data in the `data/` directory following the format specified in `data/README.md`. The analysis expects:

- Patient demographics and clinical characteristics
- Healthcare utilization history (12 months baseline)
- Acute care outcomes (12 months follow-up)
- State identifiers for domain adaptation

### 3. Configuration

Modify configuration files in `config/` to match your data and analysis requirements:

- `data_config.yaml`: Data paths, inclusion criteria, feature definitions
- `model_config.yaml`: Model hyperparameters, training settings

### 4. Run Analysis

```bash
# Full analysis pipeline
python src/main_analysis.py

# Individual components
python src/transfer_learning_models.py  # Train models
python src/evaluation_metrics.py        # Evaluate performance
python src/statistical_analysis.py     # Statistical testing
python src/calibration_analysis.py     # Calibration analysis
python src/ablation_study.py           # Ablation studies
python src/visualization.py            # Generate figures
```

## Key Features

### Transfer Learning Approaches
- Source-Only Transfer (Naive Transfer)
- Prototypical Networks for Domain Adaptation
- Model-Agnostic Meta-Learning (MAML)
- Domain Adversarial Neural Networks
- Causal Transfer Learning
- TabTransformer Architecture
- Meta-Ensemble Approach

### Evaluation Framework
- Clinical utility metrics (Youden's J Index, NNT)
- Discrimination metrics (AUC, sensitivity, specificity)
- Calibration assessment (Brier score, ECE, Hosmer-Lemeshow)
- Bootstrap confidence intervals
- Statistical significance testing with multiple comparison correction

### Analysis Components
- Comprehensive demographic and clinical characterization
- Subgroup analysis across age, gender, and comorbidity
- Feature importance and ablation studies
- Domain shift visualization and temporal stability
- Post-hoc calibration methods

## Reproducibility

### Computational Requirements
- Python 3.8+
- 16GB+ RAM recommended
- GPU optional but recommended for neural network models
- Estimated runtime: 2-4 hours for full analysis

### Random Seed Control
All random processes are seeded for reproducibility:
- Model initialization: seed=42
- Data splitting: seed=123
- Bootstrap sampling: seed=456

### Version Control
Key package versions are pinned in `requirements.txt` to ensure reproducible results.

## Extension Guidelines

### Adding New Transfer Learning Methods
1. Implement new model class in `src/transfer_learning_models.py`
2. Follow the base class interface for consistency
3. Add configuration parameters to `config/model_config.yaml`
4. Include unit tests in `tests/test_models.py`

### Adapting to New Datasets
1. Update data preprocessing in `src/data_preprocessing.py`
2. Modify feature definitions in `config/data_config.yaml`
3. Adjust evaluation metrics if needed in `src/evaluation_metrics.py`

### Adding New Evaluation Metrics
1. Implement metric functions in `src/evaluation_metrics.py`
2. Update visualization functions in `src/visualization.py`
3. Add statistical testing if appropriate in `src/statistical_analysis.py`

## Citation

If you use this code in your research, please cite:

```bibtex
@article{medicaid_transfer_learning_2024,
  title={Generalizing Healthcare Risk Prediction Models: A Prospective Evaluation of Transfer Learning for Predicting Acute Care Use in Medicaid Populations},
  author={Sanjay Basu},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions about the code or methodology, please open an issue on GitHub or contact sanjay.basu@waymarkcare.com.

