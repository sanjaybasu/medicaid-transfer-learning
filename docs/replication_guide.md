# Replication Guide

## Overview

This guide provides step-by-step instructions for replicating the transfer learning analysis for Medicaid acute care prediction. Follow these instructions to reproduce the results published in npj Health Systems.

## Prerequisites

### System Requirements
- Python 3.8 or higher
- 16GB+ RAM (32GB recommended for large datasets)
- 10GB+ available disk space
- GPU optional but recommended for neural network models

### Software Dependencies
- See `requirements.txt` for complete list
- Key packages: scikit-learn, torch, pandas, numpy, matplotlib

## Step-by-Step Replication

### Step 1: Environment Setup

#### Option A: Using Conda (Recommended)
```bash
# Clone the repository
git clone https://github.com/[username]/medicaid-transfer-learning.git
cd medicaid-transfer-learning

# Create conda environment
conda env create -f environment.yml
conda activate medicaid_transfer_learning

# Verify installation
python -c "import torch, sklearn, pandas; print('Environment setup successful')"
```

#### Option B: Using pip
```bash
# Clone the repository
git clone https://github.com/[username]/medicaid-transfer-learning.git
cd medicaid-transfer-learning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch, sklearn, pandas; print('Environment setup successful')"
```

### Step 2: Data Preparation

#### Obtain Medicaid Data
1. **Data Access**: Obtain access to Medicaid claims data from Washington and Virginia
   - Follow institutional IRB procedures
   - Establish data use agreements with state Medicaid programs
   - Ensure HIPAA compliance

2. **Data Format**: Ensure data follows the format specified in `data/README.md`
   - Required columns for demographics, clinical conditions, utilization
   - Proper coding for binary indicators and categorical variables
   - Complete outcome data for 12-month follow-up period

3. **Data Placement**: Place data files in the `data/` directory
   ```
   data/
   ├── washington_medicaid.csv
   └── virginia_medicaid.csv
   ```

#### Alternative: Synthetic Data for Testing
If real data is not available, generate synthetic data for testing:
```bash
python src/data_preprocessing.py --generate-synthetic --n-patients 10000
```

### Step 3: Configuration

#### Review Configuration Files
1. **Data Configuration** (`config/data_config.yaml`)
   - Verify data paths match your file locations
   - Adjust inclusion/exclusion criteria if needed
   - Modify feature definitions for your data structure

2. **Model Configuration** (`config/model_config.yaml`)
   - Review hyperparameters for each model
   - Adjust computational settings based on your hardware
   - Modify random seeds if desired (affects reproducibility)

#### Example Configuration Adjustments
```yaml
# In data_config.yaml
data_paths:
  source_domain: "data/your_washington_data.csv"
  target_domain: "data/your_virginia_data.csv"

# In model_config.yaml
models:
  maml:
    n_epochs: 50  # Reduce for faster testing
    inner_lr: 0.01
    outer_lr: 0.001
```

### Step 4: Run Complete Analysis

#### Full Analysis Pipeline
```bash
# Run complete analysis with default settings
python src/main_analysis.py

# Run with custom configuration
python src/main_analysis.py --config config/ --output results/

# Run with verbose logging
python src/main_analysis.py --verbose
```

#### Monitor Progress
The analysis will output progress information:
```
2024-01-15 10:00:00 - INFO - Starting Medicaid Transfer Learning Analysis
2024-01-15 10:00:01 - INFO - Step 1: Data preprocessing and quality checks
2024-01-15 10:02:15 - INFO - Step 2: Training transfer learning models
2024-01-15 10:15:30 - INFO - Step 3: Evaluating model performance
...
```

### Step 5: Individual Component Analysis

If you prefer to run components separately:

#### Data Preprocessing
```bash
python src/data_preprocessing.py --config config/data_config.yaml
```

#### Model Training
```bash
python src/transfer_learning_models.py --config config/model_config.yaml
```

#### Evaluation and Statistical Analysis
```bash
python src/evaluation_metrics.py
python src/statistical_analysis.py
python src/calibration_analysis.py
python src/ablation_study.py
```

#### Generate Visualizations
```bash
python src/visualization.py --results results/tables/
```

### Step 6: Verify Results

#### Expected Outputs
After successful completion, you should have:

1. **Tables** (in `results/tables/`)
   - `performance_model_comparison.csv`
   - `statistical_significance_tests.csv`
   - `calibration_calibration_metrics.csv`
   - `ablation_ablation_results.csv`

2. **Figures** (in `results/figures/`)
   - `figure1_clinical_utility_assessment.png`
   - `figure2_sensitivity_specificity_balance.png`
   - `figure3_performance_improvements.png`
   - Multiple supplementary figures

3. **Summary Report** (`results/analysis_summary.txt`)

#### Key Results to Verify
1. **Best Model Performance**
   - Enhanced Meta-Learning MAML should show highest AUC
   - Youden's J Index improvement of ~195%
   - NNT reduction from ~68 to ~51 patients

2. **Statistical Significance**
   - Bootstrap confidence intervals should exclude zero for top models
   - Effect sizes (Cohen's d) should be large (>0.8)

3. **Calibration Results**
   - Initial models should show poor calibration (H-L p < 0.001)
   - Post-hoc calibration should improve Brier scores

### Step 7: Troubleshooting

#### Common Issues and Solutions

1. **Memory Errors**
   ```bash
   # Reduce batch sizes in model_config.yaml
   # Use smaller synthetic dataset for testing
   python src/main_analysis.py --config config/small_config.yaml
   ```

2. **CUDA/GPU Issues**
   ```bash
   # Force CPU-only mode
   export CUDA_VISIBLE_DEVICES=""
   python src/main_analysis.py
   ```

3. **Data Format Errors**
   ```bash
   # Validate data format
   python src/data_preprocessing.py --validate-only
   ```

4. **Missing Dependencies**
   ```bash
   # Reinstall requirements
   pip install -r requirements.txt --force-reinstall
   ```

#### Performance Optimization
1. **Parallel Processing**
   - Set `n_jobs=-1` in sklearn models
   - Use GPU for neural network models
   - Consider distributed computing for large datasets

2. **Memory Management**
   - Process data in chunks for large datasets
   - Use data generators instead of loading all data
   - Monitor memory usage during execution

### Step 8: Validation and Quality Checks

#### Statistical Validation
1. **Reproducibility Check**
   ```bash
   # Run analysis twice with same random seeds
   python src/main_analysis.py --config config/
   # Results should be identical
   ```

2. **Sensitivity Analysis**
   ```bash
   # Test with different hyperparameters
   python src/main_analysis.py --config config/sensitivity_config.yaml
   ```

#### Clinical Validation
1. **Sanity Checks**
   - Verify outcome rates match expected ranges (20-40%)
   - Check feature importance aligns with clinical knowledge
   - Ensure model improvements are clinically meaningful

2. **Subgroup Analysis**
   - Verify consistent performance across demographic groups
   - Check for potential bias or disparities
   - Validate generalizability claims

## Expected Runtime

### Computational Complexity
- **Data Preprocessing**: 5-10 minutes
- **Model Training**: 30-60 minutes (depends on data size and hardware)
- **Evaluation**: 10-20 minutes
- **Statistical Analysis**: 20-30 minutes (bootstrap intensive)
- **Visualization**: 5-10 minutes
- **Total**: 1.5-2.5 hours for complete analysis

### Scaling Considerations
- Runtime scales approximately linearly with sample size
- Neural network models (MAML, TabTransformer) are most computationally intensive
- Bootstrap analysis (1000 iterations) is time-consuming but parallelizable

## Customization for Different Datasets

### Adapting to New Populations
1. **Update Feature Definitions**
   - Modify `config/data_config.yaml` for new clinical variables
   - Adjust preprocessing steps in `src/data_preprocessing.py`

2. **Adjust Model Parameters**
   - Tune hyperparameters for new data characteristics
   - Modify network architectures if needed

3. **Update Evaluation Metrics**
   - Add population-specific metrics
   - Adjust clinical significance thresholds

### Multi-State Extension
1. **Data Structure**
   - Add state identifiers to configuration
   - Modify data loading for multiple domains

2. **Model Adaptation**
   - Extend domain adaptation methods
   - Consider hierarchical modeling approaches

## Validation Against Published Results

### Key Metrics to Match
1. **Sample Characteristics**
   - Washington: 51,771 patients, 25.2% outcome rate
   - Virginia: 69,850 patients, 36.0% outcome rate

2. **Model Performance**
   - Source Only: AUC 0.512, Youden's J 0.018, NNT 67.9
   - Enhanced MAML: AUC 0.537, Youden's J 0.053, NNT 50.7

3. **Statistical Results**
   - Bootstrap 95% CI for AUC improvement: [0.001, 0.067]
   - Cohen's d effect size: 1.27 (large effect)

### Acceptable Variations
- Small numerical differences (<0.001 for AUC) due to:
  - Different random number generators
  - Floating-point precision differences
  - Software version variations

## Support and Contact

### Getting Help
1. **Check Documentation**: Review all README files and documentation
2. **Search Issues**: Check GitHub issues for similar problems
3. **Create Issue**: Open new issue with detailed error information
4. **Contact Authors**: Email research team for methodology questions

### Reporting Problems
When reporting issues, include:
- Complete error messages and stack traces
- System information (OS, Python version, package versions)
- Configuration files used
- Steps to reproduce the problem
- Expected vs. actual behavior

### Contributing Improvements
- Fork the repository
- Create feature branch
- Submit pull request with clear description
- Include tests for new functionality
- Update documentation as needed

