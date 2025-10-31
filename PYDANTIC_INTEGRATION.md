# Pydantic Integration Guide

This warranty claims ML application now uses [Pydantic](https://docs.pydantic.dev/) for data validation and configuration management.

## Overview

Pydantic provides automatic data validation, type checking, and serialization for the application's data models and configurations.

## Key Benefits

✅ **Automatic Validation** - Data is validated on input with detailed error messages
✅ **Type Safety** - Ensures data types match expectations
✅ **Value Constraints** - Enforces ranges (e.g., costs ≥ 0, vehicle year 1900-2100)
✅ **Better Documentation** - Self-documenting models with field descriptions
✅ **IDE Support** - Enhanced autocomplete and type hints

## Data Models

### 1. LaborCode Model

```python
from claim_analyzer import LaborCode
from pydantic import ValidationError

# Valid labor code
labor_code = LaborCode(
    code="2589700",
    description="Replace component A",
    performed=True,
    is_optional=False,
    labor_hours=1.5,
    labor_cost=150.0,
    parts_cost=100.0
)

# Invalid - negative cost (will raise ValidationError)
try:
    invalid_code = LaborCode(
        code="123",
        performed=True,
        labor_cost=-50.0  # ❌ Negative cost not allowed
    )
except ValidationError as e:
    print(e)
```

**Validation Rules:**
- `code`: Must not be empty or whitespace
- `labor_hours`: Must be ≥ 0
- `labor_cost`: Must be ≥ 0
- `parts_cost`: Must be ≥ 0

### 2. ClaimJob Model

```python
from claim_analyzer import ClaimJob, LaborCode

job = ClaimJob(
    job_id="JOB001",
    campaign_code="S3494",
    labor_codes=[
        LaborCode(code="LC001", description="Task 1", performed=True)
    ],
    total_labor_hours=2.5,
    total_labor_cost=250.0,
    total_parts_cost=150.0
)
```

**Validation Rules:**
- `job_id`: Must not be empty
- `campaign_code`: Must not be empty
- `labor_codes`: Must have at least one labor code
- All cost fields: Must be ≥ 0

### 3. Claim Model

```python
from claim_analyzer import Claim, ClaimJob, LaborCode

claim = Claim(
    claim_id="CLM001",
    vehicle_id="VIN12345",
    claim_date="2024-01-15",  # Auto-converted to pd.Timestamp
    dealer_id="DLR001",
    vehicle_make="Toyota",
    vehicle_model="Camry",
    vehicle_year=2020,
    mileage=35000,
    claim_jobs=[...]
)
```

**Validation Rules:**
- `claim_id`, `vehicle_id`, `dealer_id`: Must not be empty
- `vehicle_year`: Must be between 1900-2100 (if provided)
- `mileage`: Must be between 0-1,000,000 (if provided)
- `claim_jobs`: Must have at least one job

## Configuration Models

### ModelConfig

Configure machine learning models with validated parameters:

```python
from predictive_models import ModelConfig

# Default configuration
config = ModelConfig()

# Custom configuration
config = ModelConfig(
    model_type='gradient_boosting',
    n_estimators=200,
    max_depth=15,
    test_size=0.25,
    cv_folds=10
)

# Invalid configuration (will raise ValidationError)
try:
    bad_config = ModelConfig(
        model_type='invalid_model',  # ❌ Must be: random_forest, gradient_boosting, or logistic
        n_estimators=5  # ❌ Must be ≥ 10
    )
except ValidationError as e:
    print(e)
```

**Available Fields:**
- `model_type`: 'random_forest', 'gradient_boosting', or 'logistic'
- `n_estimators`: 10-1000 (default: 100)
- `max_depth`: 1-50 (default: 10)
- `random_state`: Any integer (default: 42)
- `test_size`: 0.05-0.5 (default: 0.2)
- `cv_folds`: 2-10 (default: 5)

### PredictionConfig

```python
from predictive_models import PredictionConfig

config = PredictionConfig(
    skip_rate_threshold=0.4,
    target_column='optional_skip_rate',
    min_samples_for_training=20,
    enable_feature_importance=True
)
```

### DataLoaderConfig

```python
from predictive_models import DataLoaderConfig

config = DataLoaderConfig(
    n_claims=500,
    avg_jobs_per_claim=3,
    avg_labor_codes_per_job=5,
    optional_skip_rate=0.25,
    seed=42
)
```

## Data Loading with Validation

### Load and Validate JSON

```python
from data_loader import ClaimDataLoader
from pydantic import ValidationError

loader = ClaimDataLoader()

try:
    # This validates ALL claims using Pydantic models
    claims = loader.load_and_validate_from_json('claims.json')
    print(f"✓ Successfully validated {len(claims)} claims")
except ValidationError as e:
    print(f"❌ Validation failed: {e}")
```

### Enhanced DataFrame Validation

The `validate_data()` method now includes Pydantic-inspired checks:

```python
from data_loader import ClaimDataLoader

loader = ClaimDataLoader()
df = loader.load_from_csv('claims.csv')

validation = loader.validate_data(df)

if validation['is_valid']:
    print("✓ Data is valid")
else:
    print("❌ Validation errors:")
    print(f"  Missing columns: {validation['missing_columns']}")
    print(f"  Value errors: {validation['value_errors']}")
```

**New validations include:**
- Vehicle year must be 1900-2100
- Mileage must be 0-1,000,000
- All costs/hours must be ≥ 0

## Working with Pydantic Models

### Converting to/from Dictionaries

```python
from claim_analyzer import Claim

# From dictionary
claim_dict = {
    'claim_id': 'CLM001',
    'vehicle_id': 'VIN123',
    ...
}
claim = Claim(**claim_dict)

# To dictionary
claim_dict = claim.model_dump()

# To JSON string
claim_json = claim.model_dump_json(indent=2)
```

### Accessing Properties

Pydantic models retain all the original computed properties:

```python
claim = Claim(...)

print(claim.total_cost)          # Computed property
print(claim.campaign_count)      # Number of jobs
print(claim.skip_rate)           # Overall skip rate
print(claim.optional_skip_rate)  # Optional skip rate
```

## Error Handling

Pydantic provides detailed error messages:

```python
from claim_analyzer import Claim
from pydantic import ValidationError

try:
    claim = Claim(
        claim_id='',  # Empty string
        vehicle_id='VIN123',
        claim_date='2024-01-01',
        dealer_id='DLR001',
        vehicle_year=1800,  # Too old
        mileage=-1000,  # Negative
        claim_jobs=[]  # Empty list
    )
except ValidationError as e:
    print(e.json(indent=2))
    # Shows exactly which fields failed and why:
    # - claim_id: String should have at least 1 character
    # - vehicle_year: Input should be greater than or equal to 1900
    # - mileage: Input should be greater than or equal to 0
    # - claim_jobs: List should have at least 1 item
```

## Configuration Files

You can now store configurations in JSON/YAML:

**config.json:**
```json
{
  "model_type": "random_forest",
  "n_estimators": 200,
  "max_depth": 15,
  "test_size": 0.25,
  "cv_folds": 5
}
```

**Load configuration:**
```python
from predictive_models import ModelConfig
import json

with open('config.json') as f:
    config_dict = json.load(f)

config = ModelConfig(**config_dict)  # Automatic validation!
```

## Migration Notes

### What Changed

1. **Models**: Dataclasses replaced with Pydantic BaseModel
   - ✅ All properties and methods remain the same
   - ✅ Automatic validation added
   - ✅ Better type hints

2. **ClaimAnalyzer.load_claims()**: Simplified
   - Now uses Pydantic's automatic validation
   - Fewer lines of code, more robust

3. **Data Loading**: Enhanced validation
   - New `load_and_validate_from_json()` method
   - Improved `validate_data()` with range checks

4. **Configuration**: New config classes
   - `ModelConfig`, `PredictionConfig`, `DataLoaderConfig`
   - Type-safe, validated configurations

### Backward Compatibility

✅ **All existing code continues to work!**

The Pydantic models are drop-in replacements for the dataclasses:
- Same attributes
- Same properties
- Same methods
- Just with automatic validation

## Examples

### Complete Example

```python
from claim_analyzer import ClaimAnalyzer, Claim, ClaimJob, LaborCode
from data_loader import ClaimDataLoader
from predictive_models import ModelConfig
from pydantic import ValidationError

# Create validated claim
claim = Claim(
    claim_id="CLM001",
    vehicle_id="VIN12345",
    claim_date="2024-01-15",
    dealer_id="DLR001",
    vehicle_year=2020,
    mileage=35000,
    claim_jobs=[
        ClaimJob(
            job_id="JOB001",
            campaign_code="S3494",
            labor_codes=[
                LaborCode(
                    code="2589700",
                    description="Replace component",
                    performed=True,
                    labor_hours=1.5,
                    labor_cost=150.0
                )
            ]
        )
    ]
)

# Use in analyzer
analyzer = ClaimAnalyzer()
analyzer.load_claims([claim.model_dump()])

# Get features
features_df = analyzer.create_features_dataframe()
print(features_df.head())

# Configure model with validation
try:
    config = ModelConfig(
        model_type='random_forest',
        n_estimators=200,
        test_size=0.25
    )
    print(f"✓ Model configured: {config.model_type}")
except ValidationError as e:
    print(f"❌ Invalid config: {e}")
```

## Additional Resources

- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Pydantic V2 Migration Guide](https://docs.pydantic.dev/latest/migration/)
- See `claim_analyzer.py` for model definitions
- See `predictive_models.py` for configuration models
- See `data_loader.py` for validation methods

## Summary

Pydantic integration provides:

1. ✅ **Automatic validation** - Catch errors early
2. ✅ **Type safety** - Prevent type mismatches
3. ✅ **Better DX** - IDE autocomplete and hints
4. ✅ **Self-documenting** - Field descriptions built-in
5. ✅ **Config management** - Validated, typed configs
6. ✅ **API-ready** - Easy FastAPI integration
7. ✅ **Backward compatible** - Existing code works unchanged

The application is now more robust, maintainable, and ready for production use!
