# Data Requirements and Format Specification

## Overview

This directory should contain the Medicaid claims data for both source and target domains. Due to privacy and regulatory requirements, actual patient data cannot be included in this repository. This document describes the required data format and structure.

## Data Files Expected

- `washington_medicaid.csv` - Source domain data (Washington state)
- `virginia_medicaid.csv` - Target domain data (Virginia state)

## Data Format Specification

### Required Columns

#### Patient Identifiers
- `patient_id`: Unique patient identifier (string/integer)
- `state`: State identifier ('WA' or 'VA')

#### Demographics
- `age`: Patient age in years (integer, 18-64)
- `gender`: Patient gender ('M', 'F')
- `race_ethnicity`: Race/ethnicity category ('White', 'Black', 'Hispanic', 'Other')
- `urban_rural`: Urban/rural residence indicator ('Urban', 'Rural')

#### Clinical Conditions (Binary indicators: 0/1)
- `diabetes_mellitus`: Diabetes diagnosis
- `hypertension`: Hypertension diagnosis
- `heart_disease`: Heart disease diagnosis
- `copd`: COPD diagnosis
- `mental_health_disorders`: Mental health conditions
- `substance_abuse_disorders`: Substance abuse disorders
- `charlson_comorbidity_index`: Charlson Comorbidity Index (integer, 0-10+)

#### Healthcare Utilization (12-month baseline period)
- `prior_ed_visits_12m`: Number of ED visits (integer, 0+)
- `prior_hospitalizations_12m`: Number of hospitalizations (integer, 0+)
- `prior_outpatient_visits_12m`: Number of outpatient visits (integer, 0+)
- `medication_count`: Number of unique medications (integer, 0+)

#### Medicaid Characteristics
- `eligibility_category`: Medicaid eligibility ('Disabled', 'TANF', 'Expansion', 'Other')
- `enrollment_duration`: Months of continuous enrollment (integer, 12+)
- `managed_care_enrollment`: Managed care enrollment indicator (0/1)

#### Geographic and Temporal
- `county_type`: County classification ('Metropolitan', 'Micropolitan', 'Rural')
- `health_service_area`: Health service area identifier (string)
- `enrollment_month`: Month of enrollment (1-12)
- `seasonal_indicators`: Seasonal indicator (string: 'Spring', 'Summer', 'Fall', 'Winter')

#### Outcome Variable
- `acute_care_utilization_12m`: Primary outcome - any ED visit or hospitalization in 12-month follow-up period (0/1)

### Data Quality Requirements

#### Sample Size
- Minimum 1,000 patients per state
- Recommended 10,000+ patients per state for robust analysis

#### Outcome Distribution
- Outcome rate between 5% and 95%
- Sufficient positive cases for model training (minimum 100 per state)

#### Missing Data
- Maximum 20% missing values per variable
- Complete outcome data required
- Patient ID and state must be complete

#### Temporal Requirements
- 12 months baseline period for feature calculation
- 12 months follow-up period for outcome assessment
- Continuous enrollment during both periods

### Data Preprocessing Notes

#### Inclusion Criteria Applied
- Age 18-64 years at enrollment
- 12+ months continuous Medicaid enrollment
- Prior acute care utilization in baseline period
- Enrollment in population health care management program
- Complete outcome data

#### Exclusion Criteria Applied
- Medicare dual eligibility (Medicaid-only analysis)
- Incomplete demographic data
- Missing outcome information
- Age outside 18-64 range

## Sample Data Structure

```csv
patient_id,state,age,gender,race_ethnicity,urban_rural,diabetes_mellitus,hypertension,heart_disease,copd,mental_health_disorders,substance_abuse_disorders,charlson_comorbidity_index,prior_ed_visits_12m,prior_hospitalizations_12m,prior_outpatient_visits_12m,medication_count,eligibility_category,enrollment_duration,managed_care_enrollment,county_type,health_service_area,enrollment_month,seasonal_indicators,acute_care_utilization_12m
P001,WA,45,F,White,Urban,1,1,0,0,1,0,3,2,1,12,8,Disabled,24,1,Metropolitan,HSA_001,3,Spring,1
P002,WA,32,M,Black,Rural,0,0,0,0,0,1,1,4,0,8,3,TANF,18,0,Rural,HSA_002,7,Summer,0
P003,VA,58,F,Hispanic,Urban,1,1,1,1,1,0,5,3,2,18,12,Disabled,36,1,Metropolitan,HSA_101,11,Fall,1
```

## Data Privacy and Security

### De-identification Requirements
- All patient identifiers must be de-identified
- Dates should be shifted consistently per patient
- Geographic identifiers limited to state and county type
- No free-text fields containing PHI

### Regulatory Compliance
- Data use must comply with HIPAA regulations
- IRB approval required for research use
- Data use agreements with state Medicaid programs
- Secure data handling and storage protocols

### Access Controls
- Restricted access to authorized research personnel
- Secure data transmission protocols
- Audit trails for data access and use
- Data destruction protocols after analysis completion

## Data Validation

Before running the analysis, the data preprocessing module will perform validation checks:

1. **Format Validation**: Verify column names and data types
2. **Range Validation**: Check for values outside expected ranges
3. **Completeness Validation**: Assess missing data patterns
4. **Consistency Validation**: Check for logical inconsistencies
5. **Distribution Validation**: Verify outcome rates and feature distributions

## Synthetic Data Generation

For testing and development purposes, a synthetic data generator is available:

```python
from src.data_preprocessing import SyntheticDataGenerator

generator = SyntheticDataGenerator()
synthetic_data = generator.generate_medicaid_data(
    n_patients=1000,
    outcome_rate=0.30,
    state='WA'
)
```

Note: Synthetic data should only be used for code testing and development, not for research findings or publication.

## Contact

For questions about data requirements or format specifications, please contact the research team or refer to the main repository documentation.

