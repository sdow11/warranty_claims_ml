# Prediction Systems Comparison

This repository contains **two complementary ML prediction systems** for warranty claims analysis. Here's how they compare:

## Overview

| Aspect | `predictive_models.py` (ClaimLevelPredictor) | `warranty_predictor.py` (WarrantyOperationPredictor) |
|--------|----------------------------------------------|------------------------------------------------------|
| **Type** | Multi-level hierarchical framework | Unified tiered prediction system |
| **Prediction Levels** | 3 separate models (claim/job/labor-code) | 1 adaptive model (operation-level) |
| **Integration** | Part of claim_analyzer ecosystem | Standalone CLI tool |
| **Data Source** | Pydantic-validated Claim objects | Flat CSV files |
| **Pydantic** | ✅ Fully integrated | ❌ Not used |
| **Visualizations** | ❌ None | ✅ Rich dashboards + SHAP |
| **Model Persistence** | ❌ Not implemented | ✅ Save/load with pickle |
| **CLI Interface** | ❌ Library only | ✅ Full argparse CLI |
| **SHAP Support** | ❌ Not included | ✅ Optional integration |

---

## Detailed Comparison

### 1. Architecture & Approach

#### **ClaimLevelPredictor** (Hierarchical Multi-Model)
```python
# THREE separate models for different prediction levels:

# Model 1: Claim-level
"Will this CLAIM have a high skip rate?"
Target: Binary (high/low skip rate)
Features: campaign_count, total_labor_codes, vehicle specs, etc.

# Model 2: Job-level
"Will this JOB have any skipped operations?"
Target: Binary (has skips / no skips)
Features: labor_code_count, campaign_code, cost, etc.

# Model 3: Labor-code-level
"Will this specific LABOR CODE be performed?"
Target: Binary (performed / skipped)
Features: campaign+labor code interaction, dealer, vehicle, etc.
```

**Philosophy**: Different abstraction levels need different models

#### **WarrantyOperationPredictor** (Tiered Single-Model)
```python
# ONE model with adaptive features based on available data:

# Tier 1 (Baseline - always available)
Features: historical rates, campaign patterns

# Tier 2 (+ Vehicle data)
Features: Tier 1 + vehicle_make, model, year

# Tier 3 (+ Mileage)
Features: Tier 2 + mileage brackets, vehicle age

# Tier 4 (+ Dealer/Context)
Features: Tier 3 + dealer patterns, temporal features
```

**Philosophy**: One flexible model that adapts to available data

---

### 2. Data Input Formats

#### **ClaimLevelPredictor**
```python
# Requires ClaimAnalyzer with Pydantic-validated data
from claim_analyzer import ClaimAnalyzer

analyzer = ClaimAnalyzer()
analyzer.load_claims(claims_data)  # Pydantic validation

predictor = ClaimLevelPredictor()
X, y = predictor.prepare_claim_level_data(analyzer.create_features_dataframe())
```

**Input**: Hierarchical JSON/dict with nested structure
```json
{
  "claim_id": "CLM001",
  "vehicle_id": "VIN123",
  "claim_jobs": [
    {
      "job_id": "JOB001",
      "labor_codes": [...]
    }
  ]
}
```

#### **WarrantyOperationPredictor**
```bash
# Direct CSV file input via CLI
python warranty_predictor.py --data operations.csv
python warranty_predictor.py --data ops.csv --vehicle vehicles.csv --dealer dealers.csv
```

**Input**: Flat CSV files
```csv
CLAIM_JOB_ID,SCC_CODE,LABOR_CODE,OPTIONAL_FLAG,value
CLM001-JOB001,S3494,LC2589700,1,1
```

---

### 3. Feature Engineering

#### **ClaimLevelPredictor**
**Manual feature selection per level:**

```python
# Claim-level features (hardcoded)
features = [
    'campaign_count',
    'total_labor_codes',
    'mileage',
    'vehicle_year',
    'avg_labor_codes_per_job',
    'has_multiple_campaigns',
    'high_complexity'
]

# + Encoded categoricals (dealer_id, vehicle_make, vehicle_model)
```

**Approach**: Pre-defined features, needs all data upfront

#### **WarrantyOperationPredictor**
**Automatic tiered feature engineering:**

```python
# Tier 1 (Always created)
- campaign_rate (historical performance)
- labor_rate (historical performance)
- campaign_labor_rate (interaction)
- campaign_complexity metrics
- labor_frequency

# Tier 2 (If vehicle data exists)
+ make_encoded, model_encoded, vehicle_year
+ make_campaign_rate (interaction)
+ model_campaign_rate (interaction)

# Tier 3 (If mileage exists)
+ mileage, mileage_bracket, high_mileage
+ vehicle_age

# Tier 4 (If dealer data exists)
+ dealer_encoded, dealer_rate
+ dealer_campaign_rate
+ temporal features (month, quarter)
```

**Approach**: Adaptive features based on available data

---

### 4. Model Types

#### **ClaimLevelPredictor**
```python
# Configurable via Pydantic ModelConfig
config = ModelConfig(
    model_type='random_forest',  # or 'gradient_boosting', 'logistic'
    n_estimators=100,
    max_depth=10,
    test_size=0.2,
    cv_folds=5
)

# All 3 levels use same model type
predictor = ClaimLevelPredictor(model_type='random_forest')
```

**Models**: RandomForest | GradientBoosting | Logistic (user choice)

#### **WarrantyOperationPredictor**
```python
# Fixed: GradientBoosting
model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
```

**Model**: GradientBoosting only (optimized for this task)

---

### 5. Prediction Capabilities

#### **ClaimLevelPredictor**
```python
# Hierarchical predictions at multiple levels

# 1. Claim-level: Risk score
claim_risk = predictor.predict_claim_skip_risk(X_claim)
# Output: Probability that claim has high skip rate

# 2. Job-level: Skip detection
job_has_skips = predictor.predict_job_skip_risk(X_job)
# Output: Probability that job has any skips

# 3. Labor-code-level: Performance prediction
labor_performed = predictor.predict_labor_code_performance(X_labor)
# Output: Probability that specific labor code is performed
```

**Use Case**: Multi-level risk assessment and analysis

#### **WarrantyOperationPredictor**
```python
# Single-level: Operation performance prediction
predictions, probabilities, confidence = predictor.predict(new_data)

# predictions: binary (0/1) - will operation be performed?
# probabilities: float (0-1) - confidence in prediction
# confidence: float (0-1) - prediction confidence score
```

**Use Case**: Direct operation-level predictions for scheduling/planning

---

### 6. Evaluation & Metrics

#### **ClaimLevelPredictor**
```python
results = predictor.train_claim_model(X, y)

# Results dictionary:
{
    'train_score': 0.85,
    'test_score': 0.82,
    'classification_report': '...',  # precision/recall/f1
    'confusion_matrix': [[TN, FP], [FN, TP]],
    'roc_auc': 0.87
}

# Also supports cross-validation:
cv_results = predictor.cross_validate_all_models(analyzer, cv=5)
```

**Metrics**: Accuracy, classification report, confusion matrix, ROC-AUC, CV scores

#### **WarrantyOperationPredictor**
```python
results = predictor.train(df)

# Results dictionary includes:
{
    'train_acc': 0.89,
    'test_acc': 0.86,
    'roc_auc': 0.91,
    'confusion_matrix': [[TN, FP], [FN, TP]],
    'feature_importance': DataFrame,
    'shap_values': array,  # if SHAP available
    'shap_importance': DataFrame
}
```

**Metrics**: Accuracy, ROC-AUC, confusion matrix, feature importance, SHAP values

---

### 7. Visualizations

#### **ClaimLevelPredictor**
❌ **No built-in visualizations**

Must create manually:
```python
import matplotlib.pyplot as plt

# Manual plotting required
plt.figure()
plt.bar(feature_importance['feature'], feature_importance['importance'])
plt.show()
```

#### **WarrantyOperationPredictor**
✅ **Comprehensive auto-generated visualizations**

```python
predictor.create_visualizations()

# Generates:
# 1. analysis_dashboard.png (18x12" multi-panel)
#    - Performance metrics summary
#    - Feature importance (model-based)
#    - Confusion matrix heatmap
#    - ROC curve
#    - SHAP importance
#    - Prediction distributions
#    - Recommendations panel
#
# 2. shap_analysis.png (if SHAP available)
#    - SHAP feature importance bar plot
#    - SHAP value distribution (beeswarm-style)
```

---

### 8. Feature Importance

#### **ClaimLevelPredictor**
```python
# Model-based only (if tree model)
importance = predictor.get_feature_importance(level='claim')

# Returns DataFrame:
#   feature          importance
#   campaign_count   0.234
#   mileage         0.189
#   ...
```

**Method**: Gini importance from tree models

#### **WarrantyOperationPredictor**
```python
# TWO methods automatically calculated:

# 1. Model-based (always)
model_importance = results['feature_importance']

# 2. SHAP values (if available)
shap_importance = results['shap_importance']

# SHAP provides:
# - Feature importance ranking
# - Direction of influence (positive/negative)
# - Individual prediction explanations
```

**Method**: Gini importance + SHAP values (with distribution plots)

---

### 9. Integration & Usage

#### **ClaimLevelPredictor**
**Library-style integration:**

```python
from claim_analyzer import ClaimAnalyzer
from predictive_models import ClaimLevelPredictor, ModelConfig

# 1. Load data through ClaimAnalyzer
analyzer = ClaimAnalyzer()
analyzer.load_claims(claims_data)

# 2. Configure model
config = ModelConfig(model_type='gradient_boosting')
predictor = ClaimLevelPredictor(config.model_type)

# 3. Train at multiple levels
features_df = analyzer.create_features_dataframe()
X_claim, y_claim = predictor.prepare_claim_level_data(features_df)
results = predictor.train_claim_model(X_claim, y_claim)

# 4. Make predictions
risk_scores = predictor.predict_claim_skip_risk(new_data)
```

**Best for**: Programmatic integration, analysis pipelines

#### **WarrantyOperationPredictor**
**CLI-style standalone tool:**

```bash
# 1. Simple one-line execution
python warranty_predictor.py --data operations.csv

# 2. With all data sources
python warranty_predictor.py \
    --data ops.csv \
    --vehicle vehicles.csv \
    --dealer dealers.csv \
    --output ./my_results

# 3. Skip visualizations for faster runs
python warranty_predictor.py --data ops.csv --no-viz
```

**Best for**: Production deployment, batch predictions, exploratory analysis

---

### 10. Configuration Management

#### **ClaimLevelPredictor**
✅ **Pydantic-based configuration:**

```python
from predictive_models import ModelConfig, PredictionConfig, DataLoaderConfig

# Type-safe, validated configs
model_config = ModelConfig(
    model_type='gradient_boosting',
    n_estimators=200,
    max_depth=15,
    test_size=0.25
)

prediction_config = PredictionConfig(
    skip_rate_threshold=0.4,
    min_samples_for_training=20
)

# Can save/load from JSON
config_dict = model_config.model_dump()
```

**Advantage**: Type-safe, validated, immutable configurations

#### **WarrantyOperationPredictor**
❌ **Hardcoded parameters:**

```python
# Model parameters are fixed in code
self.model = GradientBoostingClassifier(
    n_estimators=100,      # Fixed
    max_depth=5,           # Fixed
    learning_rate=0.1,     # Fixed
    random_state=42        # Fixed
)
```

**Limitation**: Must edit source code to change parameters

---

### 11. Model Persistence

#### **ClaimLevelPredictor**
❌ **Not implemented**

```python
# No save/load functionality
# Would need to implement manually
```

#### **WarrantyOperationPredictor**
✅ **Full model persistence:**

```python
# Save trained model
predictor.save_model('my_model.pkl')

# Load later
predictor = WarrantyOperationPredictor()
predictor.load_model('my_model.pkl')

# Saved components:
# - Trained model
# - Feature list
# - Label encoders
# - Tier level
```

---

## When to Use Each System

### Use **ClaimLevelPredictor** (`predictive_models.py`) when:

✅ You need **multi-level hierarchical predictions**
- Predict at claim, job, AND labor-code levels
- Different predictions for different abstraction levels

✅ You want **Pydantic validation and type safety**
- Data quality is critical
- Need validated configurations
- Part of larger validated pipeline

✅ You're doing **programmatic analysis**
- Integrating into larger Python applications
- Building custom analysis pipelines
- Need fine-grained control

✅ You want **flexible model selection**
- Try RandomForest vs GradientBoosting vs Logistic
- A/B test different model types
- Cross-validation across models

✅ You have **pre-processed hierarchical data**
- Data already in Claim → ClaimJob → LaborCode structure
- Using ClaimAnalyzer for analysis

**Example Use Cases:**
- Research and experimentation
- Multi-level risk dashboards
- Hierarchical decision support systems
- Integration with existing claim_analyzer workflows

---

### Use **WarrantyOperationPredictor** (`warranty_predictor.py`) when:

✅ You need **production-ready predictions**
- Direct CSV input
- One command execution
- Immediate results with visualizations

✅ You have **incomplete or varying data availability**
- Different data sources over time
- Want model to adapt automatically
- Progressive improvement as data grows

✅ You want **explainable predictions**
- Need SHAP analysis
- Stakeholder presentations
- Regulatory compliance

✅ You need **quick exploratory analysis**
- Fast iteration
- Visual feedback
- Minimal code

✅ You're **operationalizing predictions**
- Batch processing
- Scheduled runs
- Model versioning

**Example Use Cases:**
- Production deployment
- Scheduled batch predictions
- Business analyst workflows
- Executive reporting
- Model monitoring and drift detection

---

## Feature Comparison Matrix

| Feature | ClaimLevelPredictor | WarrantyOperationPredictor |
|---------|---------------------|----------------------------|
| **Data Validation** | ✅ Pydantic | ❌ Basic |
| **Multi-level Predictions** | ✅ 3 levels | ❌ Single level |
| **Adaptive Features** | ❌ Fixed | ✅ 4 tiers |
| **CLI Interface** | ❌ | ✅ |
| **Auto Visualizations** | ❌ | ✅ |
| **SHAP Support** | ❌ | ✅ |
| **Model Persistence** | ❌ | ✅ |
| **Config Management** | ✅ Pydantic | ❌ |
| **Cross-Validation** | ✅ | ❌ |
| **Ensemble Support** | ✅ | ❌ |
| **Historical Rate Features** | ❌ | ✅ |
| **Type Safety** | ✅ | ❌ |

---

## Could They Work Together?

**Yes! They're complementary:**

```python
# Scenario: Comprehensive warranty analysis pipeline

# 1. Use WarrantyOperationPredictor for quick initial analysis
import subprocess
subprocess.run([
    'python', 'warranty_predictor.py',
    '--data', 'operations.csv',
    '--vehicle', 'vehicles.csv'
])
# → Get immediate insights, visualizations, SHAP analysis

# 2. Load data into ClaimAnalyzer for detailed hierarchical analysis
from claim_analyzer import ClaimAnalyzer
from data_loader import ClaimDataLoader

loader = ClaimDataLoader()
df = loader.load_from_csv('operations.csv')
analyzer = ClaimAnalyzer()
analyzer.load_from_dataframe(df)

# 3. Use ClaimLevelPredictor for multi-level predictions
from predictive_models import ClaimLevelPredictor

predictor = ClaimLevelPredictor('gradient_boosting')

# Claim-level risk
X_claim, y_claim = predictor.prepare_claim_level_data(
    analyzer.create_features_dataframe()
)
claim_results = predictor.train_claim_model(X_claim, y_claim)

# Job-level predictions
X_job, y_job = predictor.prepare_job_level_data(analyzer)
job_results = predictor.train_job_model(X_job, y_job)

# Labor-code-level predictions
X_labor, y_labor = predictor.prepare_labor_code_level_data(analyzer)
labor_results = predictor.train_labor_code_model(X_labor, y_labor)

# → Get multi-level hierarchical insights with type safety
```

---

## Recommendation

**For most users starting out:**
👉 **Start with `warranty_predictor.py`**
- Fastest path to insights
- Rich visualizations
- SHAP explanations
- Production-ready

**For advanced integration:**
👉 **Migrate to `predictive_models.py`**
- When you need multi-level predictions
- When integrating into larger pipelines
- When Pydantic validation is required
- When you need configurable model types

**For production systems:**
👉 **Use BOTH**
- `warranty_predictor.py` for batch predictions and monitoring
- `predictive_models.py` for API endpoints and real-time scoring
- Share data pipeline via `data_loader.py` and `claim_analyzer.py`

---

## Summary

| Aspect | ClaimLevelPredictor | WarrantyOperationPredictor |
|--------|---------------------|----------------------------|
| **Philosophy** | Hierarchical multi-model framework | Unified adaptive prediction tool |
| **Complexity** | Higher (3 models, Pydantic, library) | Lower (1 model, CLI, standalone) |
| **Setup Time** | Longer (need claim_analyzer) | Immediate (just CSV files) |
| **Flexibility** | High (model types, configs) | Medium (tiered features) |
| **Production Ready** | Need wrappers | Yes, out of box |
| **Explainability** | Manual (feature importance) | Automatic (SHAP) |
| **Best For** | Research & Integration | Production & Exploration |

Both systems are production-quality and serve different needs. Choose based on your use case! 🚀
