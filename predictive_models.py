"""
Predictive Modeling for Claim-Level Analysis
Predicts optional labor code performance at multiple levels
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Literal
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from pydantic import BaseModel, Field, field_validator
import warnings
warnings.filterwarnings('ignore')


# Pydantic Configuration Models
class ModelConfig(BaseModel):
    """Configuration for machine learning models"""
    model_type: Literal['random_forest', 'gradient_boosting', 'logistic'] = Field(
        default='random_forest',
        description="Type of ML model to use"
    )
    n_estimators: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Number of estimators for ensemble models"
    )
    max_depth: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum depth of trees"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    test_size: float = Field(
        default=0.2,
        ge=0.05,
        le=0.5,
        description="Proportion of data for testing"
    )
    cv_folds: int = Field(
        default=5,
        ge=2,
        le=10,
        description="Number of cross-validation folds"
    )

    class Config:
        frozen = True  # Make config immutable


class PredictionConfig(BaseModel):
    """Configuration for prediction tasks"""
    skip_rate_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Threshold for high skip rate classification"
    )
    target_column: str = Field(
        default='optional_skip_rate',
        description="Target column for prediction"
    )
    min_samples_for_training: int = Field(
        default=10,
        ge=5,
        description="Minimum samples required for training"
    )
    enable_feature_importance: bool = Field(
        default=True,
        description="Whether to calculate feature importance"
    )

    class Config:
        frozen = True


class DataLoaderConfig(BaseModel):
    """Configuration for data loading and processing"""
    n_claims: int = Field(
        default=100,
        ge=1,
        le=100000,
        description="Number of claims to generate/load"
    )
    avg_jobs_per_claim: int = Field(
        default=2,
        ge=1,
        le=20,
        description="Average jobs per claim"
    )
    avg_labor_codes_per_job: int = Field(
        default=3,
        ge=1,
        le=50,
        description="Average labor codes per job"
    )
    optional_skip_rate: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Probability of skipping optional labor"
    )
    seed: int = Field(
        default=42,
        description="Random seed for data generation"
    )

    class Config:
        frozen = True


class ClaimLevelPredictor:
    """
    Multi-level prediction for warranty claims
    - Claim-level: Will this claim have high skip rate?
    - Job-level: Will this job have skipped labor codes?
    - Labor-code-level: Will this specific labor code be performed?
    """
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.claim_model = None
        self.job_model = None
        self.labor_code_model = None
        self.label_encoders = {}
        self.feature_importance = {}
        
    def _get_model(self):
        """Get model instance based on type"""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        elif self.model_type == 'logistic':
            return LogisticRegression(max_iter=1000, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def prepare_claim_level_data(self, features_df: pd.DataFrame, 
                                  target_col: str = 'optional_skip_rate',
                                  threshold: float = 0.3) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for claim-level prediction
        Target: Whether claim has high optional skip rate (binary)
        """
        df = features_df.copy()
        
        # Create binary target
        y = (df[target_col] > threshold).astype(int)
        
        # Select features
        feature_cols = [
            'campaign_count',
            'total_labor_codes',
            'mileage',
            'vehicle_year',
            'avg_labor_codes_per_job',
            'has_multiple_campaigns',
            'high_complexity'
        ]
        
        # Encode categorical features
        categorical_cols = ['dealer_id', 'vehicle_make', 'vehicle_model']
        for col in categorical_cols:
            if col in df.columns and df[col].notna().any():
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    # Fit encoder
                    valid_mask = df[col].notna()
                    self.label_encoders[col].fit(df.loc[valid_mask, col])
                
                # Transform
                valid_mask = df[col].notna()
                df.loc[valid_mask, f'{col}_encoded'] = self.label_encoders[col].transform(df.loc[valid_mask, col])
                df[f'{col}_encoded'] = df[f'{col}_encoded'].fillna(-1)
                feature_cols.append(f'{col}_encoded')
        
        X = df[feature_cols].fillna(0)
        
        return X, y
    
    def prepare_job_level_data(self, analyzer) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for job-level prediction
        Target: Whether job has any skipped optional labor codes
        """
        job_data = []
        
        for claim in analyzer.claims:
            for job in claim.claim_jobs:
                job_data.append({
                    'claim_id': claim.claim_id,
                    'job_id': job.job_id,
                    'campaign_code': job.campaign_code,
                    'dealer_id': claim.dealer_id,
                    'vehicle_make': claim.vehicle_make,
                    'vehicle_model': claim.vehicle_model,
                    'vehicle_year': claim.vehicle_year,
                    'mileage': claim.mileage,
                    'labor_code_count': len(job.labor_codes),
                    'total_labor_hours': job.total_labor_hours,
                    'total_cost': job.total_cost,
                    'claim_campaign_count': claim.campaign_count,
                    'claim_total_labor_codes': claim.total_labor_codes,
                    'has_skip': job.optional_skipped_count > 0
                })
        
        df = pd.DataFrame(job_data)
        
        # Encode categorical features
        feature_cols = [
            'labor_code_count',
            'total_labor_hours',
            'total_cost',
            'claim_campaign_count',
            'claim_total_labor_codes',
            'mileage',
            'vehicle_year'
        ]
        
        categorical_cols = ['campaign_code', 'dealer_id', 'vehicle_make', 'vehicle_model']
        for col in categorical_cols:
            if col in df.columns and df[col].notna().any():
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    valid_mask = df[col].notna()
                    self.label_encoders[col].fit(df.loc[valid_mask, col])
                
                valid_mask = df[col].notna()
                df.loc[valid_mask, f'{col}_encoded'] = self.label_encoders[col].transform(df.loc[valid_mask, col])
                df[f'{col}_encoded'] = df[f'{col}_encoded'].fillna(-1)
                feature_cols.append(f'{col}_encoded')
        
        X = df[feature_cols].fillna(0)
        y = df['has_skip'].astype(int)
        
        return X, y
    
    def prepare_labor_code_level_data(self, analyzer) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for labor-code-level prediction
        Target: Whether specific optional labor code will be performed
        """
        labor_data = []
        
        for claim in analyzer.claims:
            for job in claim.claim_jobs:
                for labor_code in job.labor_codes:
                    if labor_code.is_optional:
                        labor_data.append({
                            'claim_id': claim.claim_id,
                            'job_id': job.job_id,
                            'campaign_code': job.campaign_code,
                            'labor_code': labor_code.code,
                            'labor_description': labor_code.description,
                            'dealer_id': claim.dealer_id,
                            'vehicle_make': claim.vehicle_make,
                            'vehicle_model': claim.vehicle_model,
                            'vehicle_year': claim.vehicle_year,
                            'mileage': claim.mileage,
                            'job_labor_count': len(job.labor_codes),
                            'job_total_cost': job.total_cost,
                            'claim_campaign_count': claim.campaign_count,
                            'claim_total_cost': claim.total_cost,
                            'performed': labor_code.performed
                        })
        
        df = pd.DataFrame(labor_data)
        
        # Encode categorical features
        feature_cols = [
            'job_labor_count',
            'job_total_cost',
            'claim_campaign_count',
            'claim_total_cost',
            'mileage',
            'vehicle_year'
        ]
        
        categorical_cols = ['campaign_code', 'labor_code', 'dealer_id', 'vehicle_make', 'vehicle_model']
        for col in categorical_cols:
            if col in df.columns and df[col].notna().any():
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    valid_mask = df[col].notna()
                    self.label_encoders[col].fit(df.loc[valid_mask, col])
                
                valid_mask = df[col].notna()
                df.loc[valid_mask, f'{col}_encoded'] = self.label_encoders[col].transform(df.loc[valid_mask, col])
                df[f'{col}_encoded'] = df[f'{col}_encoded'].fillna(-1)
                feature_cols.append(f'{col}_encoded')
        
        X = df[feature_cols].fillna(0)
        y = df['performed'].astype(int)
        
        return X, y
    
    def train_claim_model(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """Train claim-level model"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        self.claim_model = self._get_model()
        self.claim_model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.claim_model.score(X_train, y_train)
        test_score = self.claim_model.score(X_test, y_test)
        
        y_pred = self.claim_model.predict(X_test)
        y_pred_proba = self.claim_model.predict_proba(X_test)[:, 1]
        
        # Feature importance
        if hasattr(self.claim_model, 'feature_importances_'):
            self.feature_importance['claim'] = pd.DataFrame({
                'feature': X.columns,
                'importance': self.claim_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        results = {
            'train_score': train_score,
            'test_score': test_score,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y)) > 1 else None
        }
        
        return results
    
    def train_job_model(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """Train job-level model"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        self.job_model = self._get_model()
        self.job_model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.job_model.score(X_train, y_train)
        test_score = self.job_model.score(X_test, y_test)
        
        y_pred = self.job_model.predict(X_test)
        y_pred_proba = self.job_model.predict_proba(X_test)[:, 1]
        
        # Feature importance
        if hasattr(self.job_model, 'feature_importances_'):
            self.feature_importance['job'] = pd.DataFrame({
                'feature': X.columns,
                'importance': self.job_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        results = {
            'train_score': train_score,
            'test_score': test_score,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y)) > 1 else None
        }
        
        return results
    
    def train_labor_code_model(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """Train labor-code-level model"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        self.labor_code_model = self._get_model()
        self.labor_code_model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.labor_code_model.score(X_train, y_train)
        test_score = self.labor_code_model.score(X_test, y_test)
        
        y_pred = self.labor_code_model.predict(X_test)
        y_pred_proba = self.labor_code_model.predict_proba(X_test)[:, 1]
        
        # Feature importance
        if hasattr(self.labor_code_model, 'feature_importances_'):
            self.feature_importance['labor_code'] = pd.DataFrame({
                'feature': X.columns,
                'importance': self.labor_code_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        results = {
            'train_score': train_score,
            'test_score': test_score,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y)) > 1 else None
        }
        
        return results
    
    def predict_claim_skip_risk(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of high skip rate for claims"""
        if self.claim_model is None:
            raise ValueError("Claim model not trained yet")
        return self.claim_model.predict_proba(X)[:, 1]
    
    def predict_job_skip_risk(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of skips in job"""
        if self.job_model is None:
            raise ValueError("Job model not trained yet")
        return self.job_model.predict_proba(X)[:, 1]
    
    def predict_labor_code_performance(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of labor code being performed"""
        if self.labor_code_model is None:
            raise ValueError("Labor code model not trained yet")
        return self.labor_code_model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self, level: str = 'all') -> Dict[str, pd.DataFrame]:
        """Get feature importance for specified level(s)"""
        if level == 'all':
            return self.feature_importance
        elif level in self.feature_importance:
            return {level: self.feature_importance[level]}
        else:
            return {}
    
    def cross_validate_all_models(self, analyzer, cv: int = 5):
        """Perform cross-validation on all three model levels"""
        results = {}
        
        # Claim level
        from claim_analyzer import ClaimAnalyzer
        features_df = analyzer.create_features_dataframe()
        X_claim, y_claim = self.prepare_claim_level_data(features_df)
        
        model = self._get_model()
        scores = cross_val_score(model, X_claim, y_claim, cv=cv, scoring='roc_auc')
        results['claim_level'] = {
            'mean_roc_auc': scores.mean(),
            'std_roc_auc': scores.std(),
            'scores': scores
        }
        
        # Job level
        X_job, y_job = self.prepare_job_level_data(analyzer)
        model = self._get_model()
        scores = cross_val_score(model, X_job, y_job, cv=cv, scoring='roc_auc')
        results['job_level'] = {
            'mean_roc_auc': scores.mean(),
            'std_roc_auc': scores.std(),
            'scores': scores
        }
        
        # Labor code level
        X_labor, y_labor = self.prepare_labor_code_level_data(analyzer)
        model = self._get_model()
        scores = cross_val_score(model, X_labor, y_labor, cv=cv, scoring='roc_auc')
        results['labor_code_level'] = {
            'mean_roc_auc': scores.mean(),
            'std_roc_auc': scores.std(),
            'scores': scores
        }
        
        return results


class EnsemblePredictor:
    """Ensemble predictor combining all three levels"""
    
    def __init__(self):
        self.claim_predictor = ClaimLevelPredictor('random_forest')
        self.job_predictor = ClaimLevelPredictor('gradient_boosting')
        self.labor_code_predictor = ClaimLevelPredictor('logistic')
    
    def train_all(self, analyzer):
        """Train all three levels"""
        results = {}
        
        # Claim level
        features_df = analyzer.create_features_dataframe()
        X_claim, y_claim = self.claim_predictor.prepare_claim_level_data(features_df)
        results['claim'] = self.claim_predictor.train_claim_model(X_claim, y_claim)
        
        # Job level
        X_job, y_job = self.job_predictor.prepare_job_level_data(analyzer)
        results['job'] = self.job_predictor.train_job_model(X_job, y_job)
        
        # Labor code level
        X_labor, y_labor = self.labor_code_predictor.prepare_labor_code_level_data(analyzer)
        results['labor_code'] = self.labor_code_predictor.train_labor_code_model(X_labor, y_labor)
        
        return results
    
    def predict_hierarchical(self, analyzer, claim_id: str) -> Dict:
        """Make hierarchical predictions for a specific claim"""
        claim = next((c for c in analyzer.claims if c.claim_id == claim_id), None)
        if claim is None:
            raise ValueError(f"Claim {claim_id} not found")
        
        # Get claim-level prediction
        features_df = analyzer.create_features_dataframe()
        claim_features = features_df[features_df['claim_id'] == claim_id]
        X_claim, _ = self.claim_predictor.prepare_claim_level_data(features_df)
        claim_features_prepared = X_claim[features_df['claim_id'] == claim_id]
        
        claim_risk = self.claim_predictor.predict_claim_skip_risk(claim_features_prepared)[0]
        
        # Get job-level predictions
        job_predictions = {}
        for job in claim.claim_jobs:
            # This would need the full job features
            pass  # Simplified for now
        
        results = {
            'claim_id': claim_id,
            'claim_level_skip_risk': claim_risk,
            'interpretation': 'High risk' if claim_risk > 0.7 else 'Moderate risk' if claim_risk > 0.3 else 'Low risk',
            'job_predictions': job_predictions
        }
        
        return results


def main():
    """Example usage"""
    from claim_analyzer import ClaimAnalyzer
    
    # Create sample data
    example_data = [
        {
            'claim_id': f'CLM{i:03d}',
            'vehicle_id': f'VIN{i:05d}',
            'claim_date': f'2024-01-{(i % 28) + 1:02d}',
            'dealer_id': f'DLR{(i % 5) + 1:03d}',
            'vehicle_make': ['Toyota', 'Honda', 'Ford'][i % 3],
            'vehicle_model': 'Model' + str((i % 3) + 1),
            'vehicle_year': 2018 + (i % 5),
            'mileage': 30000 + (i * 1000),
            'claim_jobs': [
                {
                    'job_id': f'JOB{i:03d}_1',
                    'campaign_code': ['S3494', 'S3757', 'H0018'][i % 3],
                    'total_labor_hours': 2.0 + (i % 3),
                    'total_labor_cost': 200.0 + (i % 3) * 50,
                    'total_parts_cost': 150.0,
                    'labor_codes': [
                        {
                            'code': f'LC{j:04d}',
                            'description': f'Labor {j}',
                            'performed': (i + j) % 3 != 0,
                            'is_optional': True,
                            'labor_hours': 1.0,
                            'labor_cost': 100.0,
                            'parts_cost': 50.0
                        }
                        for j in range(3)
                    ]
                }
            ]
        }
        for i in range(100)
    ]
    
    # Load and analyze
    analyzer = ClaimAnalyzer()
    analyzer.load_claims(example_data)
    
    print("=== Training Multi-Level Models ===")
    predictor = ClaimLevelPredictor('random_forest')
    
    # Claim level
    features_df = analyzer.create_features_dataframe()
    X_claim, y_claim = predictor.prepare_claim_level_data(features_df)
    print(f"\nClaim Level: {len(X_claim)} samples, {X_claim.shape[1]} features")
    results_claim = predictor.train_claim_model(X_claim, y_claim)
    print(f"Test ROC-AUC: {results_claim['roc_auc']:.3f}")
    
    # Job level
    X_job, y_job = predictor.prepare_job_level_data(analyzer)
    print(f"\nJob Level: {len(X_job)} samples, {X_job.shape[1]} features")
    results_job = predictor.train_job_model(X_job, y_job)
    print(f"Test ROC-AUC: {results_job['roc_auc']:.3f}")
    
    # Labor code level
    X_labor, y_labor = predictor.prepare_labor_code_level_data(analyzer)
    print(f"\nLabor Code Level: {len(X_labor)} samples, {X_labor.shape[1]} features")
    results_labor = predictor.train_labor_code_model(X_labor, y_labor)
    print(f"Test ROC-AUC: {results_labor['roc_auc']:.3f}")
    
    # Feature importance
    print("\n=== Top Features by Level ===")
    for level, importance_df in predictor.get_feature_importance().items():
        print(f"\n{level.upper()}:")
        print(importance_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
