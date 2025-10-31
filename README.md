# Warranty Claims Analysis System

A comprehensive multi-level analysis and prediction system for automotive warranty claims with full hierarchical modeling.

## System Architecture

### Three-Level Hierarchy

```
Claim (One Vehicle Visit)
  ├── Claim Job 1 (Campaign S3494)
  │   ├── Labor Code 2589700 (Required, Performed)
  │   ├── Labor Code 2557300 (Optional, Performed)
  │   └── Labor Code 1702800 (Optional, Not Performed)
  ├── Claim Job 2 (Campaign S3757)
  │   └── Labor Code 2851300 (Optional, Performed)
  └── Claim Job 3 (Campaign H0018)
      └── Labor Code 3810400 (Optional, Not Performed)
```

## Modules

### 1. claim_analyzer.py
**Core data structures and analysis**

- `LaborCode`: Individual labor operation
- `ClaimJob`: Campaign/job within a claim
- `Claim`: Complete claim for one vehicle visit
- `ClaimAnalyzer`: Comprehensive claim-level analysis

**Key Features:**
- Load claims from structured data or flat DataFrames
- Extract comprehensive features at all hierarchy levels
- Analyze skip patterns for optional labor codes
- Dealer-specific pattern analysis
- Campaign combination analysis
- Summary statistics

**Usage:**
```python
from claim_analyzer import ClaimAnalyzer

# Load data
analyzer = ClaimAnalyzer()
analyzer.load_from_dataframe(df)

# Get features
features_df = analyzer.create_features_dataframe()

# Analyze patterns
skip_patterns = analyzer.analyze_skip_patterns()
dealer_patterns = analyzer.analyze_dealer_patterns()
campaign_combos = analyzer.analyze_campaign_combinations()

# Summary
summary = analyzer.get_summary_statistics()
```

### 2. predictive_models.py
**Multi-level predictive modeling**

Three prediction levels:
1. **Claim-level**: Will this claim have a high skip rate?
2. **Job-level**: Will this job have skipped labor codes?
3. **Labor-code-level**: Will this specific labor code be performed?

**Models Supported:**
- Random Forest (default)
- Gradient Boosting
- Logistic Regression

**Usage:**
```python
from predictive_models import ClaimLevelPredictor, EnsemblePredictor

# Single-level prediction
predictor = ClaimLevelPredictor('random_forest')

# Claim level
X, y = predictor.prepare_claim_level_data(features_df)
results = predictor.train_claim_model(X, y)

# Job level
X_job, y_job = predictor.prepare_job_level_data(analyzer)
results_job = predictor.train_job_model(X_job, y_job)

# Labor code level
X_labor, y_labor = predictor.prepare_labor_code_level_data(analyzer)
results_labor = predictor.train_labor_code_model(X_labor, y_labor)

# Ensemble approach
ensemble = EnsemblePredictor()
all_results = ensemble.train_all(analyzer)
```

### 3. data_loader.py
**Data loading and preprocessing utilities**

**Supported Formats:**
- CSV (flat or hierarchical)
- JSON (structured)
- Excel
- Synthetic data generation

**Key Features:**
- Data validation
- Derived feature engineering
- Time-based splitting for validation
- Campaign/dealer filtering
- Export for ML training

**Usage:**
```python
from data_loader import ClaimDataLoader

loader = ClaimDataLoader()

# Load from CSV
df = loader.load_from_csv('claims.csv')

# Generate synthetic data for testing
df = loader.create_synthetic_data(
    n_claims=1000,
    avg_jobs_per_claim=2,
    avg_labor_codes_per_job=3,
    optional_skip_rate=0.3
)

# Validate data
validation = loader.validate_data(df)

# Add derived features
df_enhanced = loader.add_derived_features(df)

# Split by date for time-series validation
splits = loader.split_by_date(df, train_end_date='2024-09-30')

# Export for ML
loader.export_for_ml(df_enhanced, 'processed_claims.csv')
```

### 4. visualizations.py
**Comprehensive visualization suite**

**Visualization Types:**
- Skip rate distributions
- Campaign analysis (skip rates, costs, frequencies)
- Dealer comparisons
- Temporal trends
- Feature importance plots
- Confusion matrices
- Labor code skip patterns
- Executive summary dashboard

**Usage:**
```python
from visualizations import ClaimVisualizer

viz = ClaimVisualizer(output_dir='./figures')

# Create all visualizations
viz.plot_skip_rate_distribution(analyzer)
viz.plot_campaign_analysis(analyzer)

dealer_stats = analyzer.analyze_dealer_patterns()
viz.plot_dealer_comparison(dealer_stats)

viz.plot_temporal_trends(analyzer)

skip_patterns = analyzer.analyze_skip_patterns()
viz.plot_labor_code_analysis(skip_patterns)

viz.create_executive_summary(analyzer)
```

## Complete Workflow Example

```python
from claim_analyzer import ClaimAnalyzer
from predictive_models import ClaimLevelPredictor
from data_loader import ClaimDataLoader
from visualizations import ClaimVisualizer

# 1. Load data
loader = ClaimDataLoader()
df = loader.load_from_csv('warranty_claims.csv')
df = loader.add_derived_features(df)

# 2. Validate
validation = loader.validate_data(df)
print(f"Data valid: {validation['is_valid']}")

# 3. Analyze
analyzer = ClaimAnalyzer()
analyzer.load_from_dataframe(df)

print("\n=== Summary Statistics ===")
summary = analyzer.get_summary_statistics()
for key, value in summary.items():
    print(f"{key}: {value}")

# 4. Train models
predictor = ClaimLevelPredictor('random_forest')

# Claim level
features_df = analyzer.create_features_dataframe()
X_claim, y_claim = predictor.prepare_claim_level_data(features_df)
results_claim = predictor.train_claim_model(X_claim, y_claim)
print(f"\nClaim Model ROC-AUC: {results_claim['roc_auc']:.3f}")

# Job level
X_job, y_job = predictor.prepare_job_level_data(analyzer)
results_job = predictor.train_job_model(X_job, y_job)
print(f"Job Model ROC-AUC: {results_job['roc_auc']:.3f}")

# Labor code level
X_labor, y_labor = predictor.prepare_labor_code_level_data(analyzer)
results_labor = predictor.train_labor_code_model(X_labor, y_labor)
print(f"Labor Code Model ROC-AUC: {results_labor['roc_auc']:.3f}")

# 5. Visualize
viz = ClaimVisualizer()
viz.create_executive_summary(analyzer)
viz.plot_feature_importance(predictor.get_feature_importance())

# 6. Get insights
skip_patterns = analyzer.analyze_skip_patterns()
print("\n=== Top Skipped Labor Codes ===")
print(skip_patterns.head(10))

dealer_patterns = analyzer.analyze_dealer_patterns()
print("\n=== Dealers with Highest Skip Rates ===")
print(dealer_patterns.head(10))
```

## Data Format Requirements

### CSV Input Format

Required columns:
- `claim_id`: Unique claim identifier
- `vehicle_id`: Vehicle identifier
- `claim_date`: Date of claim
- `dealer_id`: Dealer identifier
- `job_id`: Job instance identifier
- `campaign_code`: Campaign code (e.g., S3494)
- `labor_code`: Labor operation code
- `performed`: Boolean (True/False)

Optional columns:
- `labor_description`: Description of labor
- `is_optional`: Boolean (default: True)
- `labor_hours`: Hours spent
- `labor_cost`: Labor cost
- `parts_cost`: Parts cost
- `vehicle_make`: Vehicle manufacturer
- `vehicle_model`: Vehicle model
- `vehicle_year`: Model year
- `mileage`: Vehicle mileage

### JSON Input Format

```json
[
  {
    "claim_id": "CLM001",
    "vehicle_id": "VIN12345",
    "claim_date": "2024-01-15",
    "dealer_id": "DLR001",
    "vehicle_make": "Toyota",
    "vehicle_model": "Camry",
    "vehicle_year": 2020,
    "mileage": 35000,
    "claim_jobs": [
      {
        "job_id": "JOB001",
        "campaign_code": "S3494",
        "total_labor_hours": 2.5,
        "total_labor_cost": 250.0,
        "total_parts_cost": 150.0,
        "labor_codes": [
          {
            "code": "2589700",
            "description": "Replace component A",
            "performed": true,
            "is_optional": false,
            "labor_hours": 1.5,
            "labor_cost": 150.0,
            "parts_cost": 100.0
          }
        ]
      }
    ]
  }
]
```

## Key Metrics

### Claim-Level Metrics
- **skip_rate**: Overall labor code skip rate
- **optional_skip_rate**: Skip rate for optional labor only
- **campaign_count**: Number of campaigns in claim
- **total_labor_codes**: Total labor operations
- **total_cost**: Total claim cost
- **complexity indicators**: Multiple campaigns, high labor code count

### Job-Level Metrics
- **performed_count**: Number of performed labor codes
- **skipped_count**: Number of skipped labor codes
- **optional_performed_count**: Optional labor performed
- **optional_skipped_count**: Optional labor skipped
- **total_cost**: Job total cost

### Labor-Code-Level Metrics
- **performed**: Boolean performance status
- **labor_hours**: Hours if performed
- **labor_cost**: Labor cost
- **parts_cost**: Parts cost

## Feature Engineering

The system automatically generates:

**Temporal Features:**
- claim_year, claim_month, claim_quarter
- claim_dayofweek, claim_is_weekend
- vehicle_age (calculated from claim_date and vehicle_year)

**Aggregate Features:**
- Claims per dealer
- Average labor codes per job
- Campaign combinations
- Cost variance across jobs

**Complexity Indicators:**
- has_multiple_campaigns
- high_complexity (>10 labor codes)
- mixed_performance (some performed, some skipped)

## Model Outputs

### Classification Reports
- Precision, Recall, F1-score for each class
- Support (sample counts)

### Feature Importance
- Top contributing features at each level
- Ranked by importance score

### Confusion Matrices
- True positives, false positives
- True negatives, false negatives

### ROC-AUC Scores
- Model discrimination ability
- Threshold-independent performance metric

## Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Testing

Each module includes a `main()` function for testing:

```bash
# Test individual modules
python claim_analyzer.py
python predictive_models.py
python data_loader.py
python visualizations.py

# Run complete workflow
python run_analysis.py
```

## Output

All outputs are saved to `./minimal_warranty_ml_results/`:
- `figures/`: Visualization PNG files
- `sample_data.csv`: Generated or processed data
- Model objects and feature importance reports

## Performance Considerations

- **Claim-level models**: Fast, suitable for executive dashboards
- **Job-level models**: Moderate complexity, good for campaign analysis
- **Labor-code-level models**: Most detailed, highest computational cost

## Future Enhancements

1. Deep learning models for sequential patterns
2. Real-time prediction APIs
3. Dealer-specific model fine-tuning
4. Cost optimization recommendations
5. Anomaly detection for unusual skip patterns
6. Integration with warranty management systems

## Support

For questions or issues, refer to module docstrings or run:
```python
help(ClaimAnalyzer)
help(ClaimLevelPredictor)
help(ClaimDataLoader)
help(ClaimVisualizer)
```
