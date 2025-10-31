# Quick Reference Card

## 🎯 Core Commands

```bash
# Quick test (5 sec)
python run_analysis.py --mode quick --claims 50

# Full analysis (30 sec)
python run_analysis.py --mode full --claims 500

# No visualizations
python run_analysis.py --claims 1000 --no-viz

# No models
python run_analysis.py --claims 200 --no-models
```

## 📊 Core Imports

```python
from claim_analyzer import ClaimAnalyzer
from predictive_models import ClaimLevelPredictor
from data_loader import ClaimDataLoader
from visualizations import ClaimVisualizer
```

## 🔄 Basic Workflow

```python
# 1. Load data
loader = ClaimDataLoader()
df = loader.load_from_csv('claims.csv')

# 2. Analyze
analyzer = ClaimAnalyzer()
analyzer.load_from_dataframe(df)
summary = analyzer.get_summary_statistics()

# 3. Train models
predictor = ClaimLevelPredictor('random_forest')
X, y = predictor.prepare_claim_level_data(features_df)
results = predictor.train_claim_model(X, y)

# 4. Visualize
viz = ClaimVisualizer()
viz.create_executive_summary(analyzer)
```

## 📁 File Guide

| File | Lines | Purpose |
|------|-------|---------|
| `claim_analyzer.py` | 477 | Data structures & analysis |
| `predictive_models.py` | 501 | ML models (3 levels) |
| `data_loader.py` | 381 | Data pipeline |
| `visualizations.py` | 427 | Plotting suite |
| `run_analysis.py` | 326 | Orchestration |
| `README.md` | 419 | API docs |
| `PROJECT_OVERVIEW.md` | 321 | System overview |
| `SUMMARY.md` | - | Complete summary |

## 🎨 Visualization Types

1. `plot_skip_rate_distribution()` - Skip rate histogram + box plot
2. `plot_campaign_analysis()` - 4 campaign charts
3. `plot_dealer_comparison()` - 4 dealer plots
4. `plot_temporal_trends()` - 4 time series
5. `plot_labor_code_analysis()` - Top skipped codes
6. `plot_feature_importance()` - ML features
7. `plot_confusion_matrices()` - Model performance
8. `create_executive_summary()` - Dashboard

## 🔍 Key Analysis Methods

```python
# Skip patterns
skip_patterns = analyzer.analyze_skip_patterns()

# Dealer patterns
dealer_stats = analyzer.analyze_dealer_patterns()

# Campaign combinations
combos = analyzer.analyze_campaign_combinations()

# Summary statistics
summary = analyzer.get_summary_statistics()

# Features for ML
features_df = analyzer.create_features_dataframe()
```

## 🤖 Model Training

```python
predictor = ClaimLevelPredictor('random_forest')

# Claim level
X_claim, y_claim = predictor.prepare_claim_level_data(features_df)
results = predictor.train_claim_model(X_claim, y_claim)
print(f"ROC-AUC: {results['roc_auc']:.3f}")

# Job level
X_job, y_job = predictor.prepare_job_level_data(analyzer)
predictor.train_job_model(X_job, y_job)

# Labor code level
X_labor, y_labor = predictor.prepare_labor_code_level_data(analyzer)
predictor.train_labor_code_model(X_labor, y_labor)

# Feature importance
importance = predictor.get_feature_importance()
```

## 📊 Data Format

### Required Columns
- `claim_id` - Unique claim identifier
- `vehicle_id` - Vehicle identifier
- `claim_date` - Date of claim
- `dealer_id` - Dealer identifier
- `job_id` - Job identifier
- `campaign_code` - Campaign code
- `labor_code` - Labor operation code
- `performed` - Boolean (True/False)

### Optional Columns
- `is_optional` - Boolean (default True)
- `labor_hours` - Float
- `labor_cost` - Float
- `parts_cost` - Float
- `vehicle_make`, `vehicle_model`, `vehicle_year`, `mileage`

## 🎲 Generate Test Data

```python
loader = ClaimDataLoader()
df = loader.create_synthetic_data(
    n_claims=500,
    avg_jobs_per_claim=2,
    avg_labor_codes_per_job=3,
    optional_skip_rate=0.3
)
```

## 📈 Key Metrics

```python
summary = analyzer.get_summary_statistics()

# Available metrics:
summary['total_claims']
summary['overall_skip_rate']
summary['overall_optional_skip_rate']
summary['total_cost']
summary['avg_cost_per_claim']
summary['avg_campaigns_per_claim']
summary['unique_dealers']
summary['unique_campaigns']
```

## 🔧 Troubleshooting

```python
# Validate data
validation = loader.validate_data(df)
print(validation['is_valid'])
print(validation['warnings'])

# Check data quality
summary = loader.get_data_summary(df)

# Add features
df = loader.add_derived_features(df)
```

## 💾 Save Results

```python
# Export processed data
loader.export_for_ml(df, 'processed_claims.csv')

# Save visualizations (automatic)
viz = ClaimVisualizer(output_dir='./my_figures')
viz.create_executive_summary(analyzer)  # Saves to my_figures/
```

## ⚡ Performance Tips

- Use `--mode quick` for fast testing
- Skip viz with `--no-viz` for faster runs
- Skip models with `--no-models` for analysis only
- Start with small datasets (<100 claims) for testing
- Scale up gradually to 500+ claims

## 📞 Getting Help

```python
help(ClaimAnalyzer)
help(ClaimLevelPredictor)
help(ClaimDataLoader)
help(ClaimVisualizer)
```

## 🎯 Common Tasks

### Task 1: Analyze existing data
```python
df = pd.read_csv('my_claims.csv')
analyzer = ClaimAnalyzer()
analyzer.load_from_dataframe(df)
skip_patterns = analyzer.analyze_skip_patterns()
```

### Task 2: Train models
```python
features_df = analyzer.create_features_dataframe()
predictor = ClaimLevelPredictor()
X, y = predictor.prepare_claim_level_data(features_df)
results = predictor.train_claim_model(X, y)
```

### Task 3: Create visualizations
```python
viz = ClaimVisualizer()
viz.plot_skip_rate_distribution(analyzer)
viz.create_executive_summary(analyzer)
```

### Task 4: Find problematic dealers
```python
dealer_stats = analyzer.analyze_dealer_patterns()
print(dealer_stats.head(10))  # Top 10 worst dealers
```

### Task 5: Identify high-skip labor codes
```python
skip_patterns = analyzer.analyze_skip_patterns()
print(skip_patterns.head(20))  # Top 20 skipped codes
```

---

**Quick Start:** `python run_analysis.py --mode quick --claims 50`
