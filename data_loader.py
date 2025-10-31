"""
Data Loading Utilities for Warranty Claims
Handles various input formats and data transformations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
import json
from pydantic import ValidationError


class ClaimDataLoader:
    """Load warranty claim data from various formats"""
    
    @staticmethod
    def load_from_csv(filepath: Union[str, Path],
                     claim_id_col: str = 'claim_id',
                     vehicle_id_col: str = 'vehicle_id',
                     date_col: str = 'claim_date',
                     dealer_col: str = 'dealer_id',
                     job_id_col: str = 'job_id',
                     campaign_col: str = 'campaign_code',
                     labor_code_col: str = 'labor_code',
                     performed_col: str = 'performed',
                     optional_col: str = 'is_optional') -> pd.DataFrame:
        """
        Load flat CSV file with warranty claim data
        
        Expected columns:
        - claim_id: Unique identifier for the claim
        - vehicle_id: Vehicle identifier
        - claim_date: Date of claim
        - dealer_id: Dealer who processed the claim
        - job_id: Job/campaign instance identifier
        - campaign_code: Campaign code (e.g., S3494)
        - labor_code: Labor operation code
        - labor_description: Description of labor
        - performed: Boolean indicating if labor was performed
        - is_optional: Boolean indicating if labor is optional
        - labor_hours: Hours spent (if performed)
        - labor_cost: Labor cost (if performed)
        - parts_cost: Parts cost (if performed)
        - vehicle_make, vehicle_model, vehicle_year, mileage (optional)
        """
        df = pd.read_csv(filepath)
        
        # Rename columns to standard names if needed
        rename_map = {
            claim_id_col: 'claim_id',
            vehicle_id_col: 'vehicle_id',
            date_col: 'claim_date',
            dealer_col: 'dealer_id',
            job_id_col: 'job_id',
            campaign_col: 'campaign_code',
            labor_code_col: 'labor_code',
            performed_col: 'performed',
            optional_col: 'is_optional'
        }
        df = df.rename(columns=rename_map)
        
        # Convert data types
        df['claim_date'] = pd.to_datetime(df['claim_date'])
        df['performed'] = df['performed'].astype(bool)
        df['is_optional'] = df['is_optional'].fillna(True).astype(bool)
        
        # Fill numeric columns
        numeric_cols = ['labor_hours', 'labor_cost', 'parts_cost', 'mileage']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(float)
        
        return df
    
    @staticmethod
    def load_from_json(filepath: Union[str, Path]) -> List[Dict]:
        """
        Load claims from JSON file
        Expected structure: List of claim dictionaries with nested jobs and labor codes
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data

    @staticmethod
    def load_and_validate_from_json(filepath: Union[str, Path]) -> List[Dict]:
        """
        Load claims from JSON file with Pydantic validation

        This method validates the entire claim structure including:
        - Required fields presence
        - Data types
        - Value ranges
        - Nested structure integrity

        Returns:
            List of validated claim dictionaries

        Raises:
            ValidationError: If data doesn't meet validation requirements
        """
        from claim_analyzer import Claim

        with open(filepath, 'r') as f:
            data = json.load(f)

        # Validate each claim using Pydantic
        validated_claims = []
        errors = []

        for idx, claim_dict in enumerate(data):
            try:
                # Pydantic validation happens here
                claim = Claim(**claim_dict)
                # Convert back to dict for compatibility
                validated_claims.append(claim.model_dump())
            except ValidationError as e:
                errors.append({
                    'claim_index': idx,
                    'claim_id': claim_dict.get('claim_id', 'unknown'),
                    'errors': e.errors()
                })

        if errors:
            print(f"Validation errors found in {len(errors)} claims:")
            for error in errors:
                print(f"  Claim {error['claim_id']} (index {error['claim_index']}): {error['errors']}")
            raise ValueError(f"Failed to validate {len(errors)} claims. See errors above.")

        print(f"Successfully validated {len(validated_claims)} claims")
        return validated_claims
    
    @staticmethod
    def load_from_excel(filepath: Union[str, Path],
                       sheet_name: str = 'Claims') -> pd.DataFrame:
        """Load from Excel file"""
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        return df
    
    @staticmethod
    def validate_data(df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate loaded data and return statistics

        This is a basic DataFrame-level validation. For comprehensive validation
        including value ranges, nested structures, and business rules, use
        load_and_validate_from_json() with Pydantic models.
        """
        required_cols = [
            'claim_id', 'vehicle_id', 'claim_date', 'dealer_id',
            'job_id', 'campaign_code', 'labor_code', 'performed'
        ]

        validation = {
            'is_valid': True,
            'missing_columns': [],
            'null_counts': {},
            'data_types': {},
            'warnings': [],
            'value_errors': []
        }

        # Check required columns
        for col in required_cols:
            if col not in df.columns:
                validation['missing_columns'].append(col)
                validation['is_valid'] = False

        if not validation['is_valid']:
            return validation

        # Check for nulls
        for col in required_cols:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                validation['null_counts'][col] = null_count
                validation['warnings'].append(f"{col} has {null_count} null values")

        # Check data types
        validation['data_types'] = {col: str(df[col].dtype) for col in df.columns}

        # Check for data quality issues
        if 'performed' in df.columns:
            if not df['performed'].dtype == bool:
                validation['warnings'].append("'performed' column should be boolean")

        # Pydantic-inspired value range checks
        if 'vehicle_year' in df.columns:
            invalid_years = df[(df['vehicle_year'] < 1900) | (df['vehicle_year'] > 2100)]['vehicle_year'].dropna()
            if len(invalid_years) > 0:
                validation['value_errors'].append(
                    f"Found {len(invalid_years)} invalid vehicle years (must be 1900-2100)"
                )
                validation['is_valid'] = False

        if 'mileage' in df.columns:
            invalid_mileage = df[(df['mileage'] < 0) | (df['mileage'] > 1_000_000)]['mileage'].dropna()
            if len(invalid_mileage) > 0:
                validation['value_errors'].append(
                    f"Found {len(invalid_mileage)} invalid mileage values (must be 0-1,000,000)"
                )
                validation['is_valid'] = False

        # Check for negative costs
        cost_cols = ['labor_hours', 'labor_cost', 'parts_cost']
        for col in cost_cols:
            if col in df.columns:
                negative_values = df[df[col] < 0][col].dropna()
                if len(negative_values) > 0:
                    validation['value_errors'].append(
                        f"Found {len(negative_values)} negative values in {col} (must be >= 0)"
                    )
                    validation['is_valid'] = False

        # Summary statistics
        validation['summary'] = {
            'total_rows': len(df),
            'unique_claims': df['claim_id'].nunique(),
            'unique_vehicles': df['vehicle_id'].nunique(),
            'unique_dealers': df['dealer_id'].nunique(),
            'unique_campaigns': df['campaign_code'].nunique(),
            'date_range': (df['claim_date'].min(), df['claim_date'].max())
        }

        return validation
    
    @staticmethod
    def create_synthetic_data(n_claims: int = 100,
                            avg_jobs_per_claim: int = 2,
                            avg_labor_codes_per_job: int = 3,
                            optional_skip_rate: float = 0.3,
                            seed: int = 42) -> pd.DataFrame:
        """
        Generate synthetic warranty claim data for testing
        
        Args:
            n_claims: Number of claims to generate
            avg_jobs_per_claim: Average jobs per claim
            avg_labor_codes_per_job: Average labor codes per job
            optional_skip_rate: Probability of skipping optional labor
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        
        data = []
        
        campaigns = ['S3494', 'S3757', 'H0018', 'T2589', 'R4821']
        makes = ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'Nissan']
        models = ['Sedan', 'SUV', 'Truck', 'Coupe', 'Hatchback']
        
        for claim_idx in range(n_claims):
            claim_id = f'CLM{claim_idx:05d}'
            vehicle_id = f'VIN{claim_idx:07d}'
            claim_date = pd.Timestamp('2024-01-01') + pd.Timedelta(days=np.random.randint(0, 365))
            dealer_id = f'DLR{np.random.randint(1, 21):03d}'
            vehicle_make = np.random.choice(makes)
            vehicle_model = np.random.choice(models)
            vehicle_year = np.random.randint(2018, 2024)
            mileage = np.random.randint(10000, 100000)
            
            # Generate jobs for this claim
            n_jobs = np.random.poisson(avg_jobs_per_claim) + 1
            
            for job_idx in range(n_jobs):
                job_id = f'{claim_id}_JOB{job_idx:02d}'
                campaign_code = np.random.choice(campaigns)
                
                # Generate labor codes for this job
                n_labor_codes = np.random.poisson(avg_labor_codes_per_job) + 1
                
                for lc_idx in range(n_labor_codes):
                    labor_code = f'LC{np.random.randint(1000000, 9999999)}'
                    labor_description = f'Labor operation {labor_code}'
                    is_optional = np.random.random() < 0.7  # 70% are optional
                    
                    # Determine if performed
                    if is_optional:
                        # Optional labor has skip probability
                        performed = np.random.random() > optional_skip_rate
                    else:
                        # Required labor almost always performed
                        performed = np.random.random() > 0.05
                    
                    # Generate costs if performed
                    labor_hours = np.random.uniform(0.5, 4.0) if performed else 0.0
                    labor_cost = labor_hours * np.random.uniform(80, 120) if performed else 0.0
                    parts_cost = np.random.uniform(50, 500) if performed and np.random.random() > 0.3 else 0.0
                    
                    data.append({
                        'claim_id': claim_id,
                        'vehicle_id': vehicle_id,
                        'claim_date': claim_date,
                        'dealer_id': dealer_id,
                        'vehicle_make': vehicle_make,
                        'vehicle_model': vehicle_model,
                        'vehicle_year': vehicle_year,
                        'mileage': mileage,
                        'job_id': job_id,
                        'campaign_code': campaign_code,
                        'labor_code': labor_code,
                        'labor_description': labor_description,
                        'performed': performed,
                        'is_optional': is_optional,
                        'labor_hours': labor_hours,
                        'labor_cost': labor_cost,
                        'parts_cost': parts_cost
                    })
        
        df = pd.DataFrame(data)
        return df
    
    @staticmethod
    def split_by_date(df: pd.DataFrame,
                     train_end_date: str,
                     val_end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Split data by date for time-series validation
        
        Args:
            df: DataFrame with claim_date column
            train_end_date: End date for training data (inclusive)
            val_end_date: End date for validation data (inclusive). If None, uses all remaining data for test
        
        Returns:
            Dictionary with 'train', 'val', and 'test' DataFrames
        """
        train_end = pd.to_datetime(train_end_date)
        
        train_df = df[df['claim_date'] <= train_end].copy()
        
        if val_end_date:
            val_end = pd.to_datetime(val_end_date)
            val_df = df[(df['claim_date'] > train_end) & (df['claim_date'] <= val_end)].copy()
            test_df = df[df['claim_date'] > val_end].copy()
        else:
            val_df = df[df['claim_date'] > train_end].copy()
            test_df = pd.DataFrame()
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
    
    @staticmethod
    def filter_by_campaign(df: pd.DataFrame, campaigns: List[str]) -> pd.DataFrame:
        """Filter data to specific campaigns"""
        return df[df['campaign_code'].isin(campaigns)].copy()
    
    @staticmethod
    def filter_by_dealer(df: pd.DataFrame, dealers: List[str]) -> pd.DataFrame:
        """Filter data to specific dealers"""
        return df[df['dealer_id'].isin(dealers)].copy()
    
    @staticmethod
    def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to the dataframe"""
        df = df.copy()
        
        # Time-based features
        if 'claim_date' in df.columns:
            df['claim_year'] = df['claim_date'].dt.year
            df['claim_month'] = df['claim_date'].dt.month
            df['claim_quarter'] = df['claim_date'].dt.quarter
            df['claim_dayofweek'] = df['claim_date'].dt.dayofweek
            df['claim_is_weekend'] = df['claim_dayofweek'].isin([5, 6])
        
        # Vehicle age
        if 'vehicle_year' in df.columns and 'claim_date' in df.columns:
            df['vehicle_age'] = df['claim_date'].dt.year - df['vehicle_year']
        
        # Cost features
        if all(col in df.columns for col in ['labor_cost', 'parts_cost']):
            df['total_cost'] = df['labor_cost'] + df['parts_cost']
        
        # Aggregate features at claim level
        claim_agg = df.groupby('claim_id').agg({
            'job_id': 'nunique',
            'labor_code': 'count'
        }).rename(columns={'job_id': 'n_jobs', 'labor_code': 'n_labor_codes'})
        
        df = df.merge(claim_agg, left_on='claim_id', right_index=True, how='left')
        
        return df
    
    @staticmethod
    def export_for_ml(df: pd.DataFrame, output_path: Union[str, Path]) -> None:
        """Export processed data for ML training"""
        df.to_csv(output_path, index=False)
        print(f"Exported {len(df)} rows to {output_path}")
    
    @staticmethod
    def get_data_summary(df: pd.DataFrame) -> Dict[str, any]:
        """Get comprehensive data summary"""
        summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'null_counts': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'numeric_summary': df.describe().to_dict(),
            'categorical_summary': {}
        }
        
        # Categorical summaries
        categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
        for col in categorical_cols:
            value_counts = df[col].value_counts().head(10)
            summary['categorical_summary'][col] = {
                'unique_count': df[col].nunique(),
                'top_values': value_counts.to_dict()
            }
        
        return summary


def main():
    """Example usage"""
    print("=== Creating Synthetic Data ===")
    loader = ClaimDataLoader()
    
    # Generate synthetic data
    df = loader.create_synthetic_data(
        n_claims=200,
        avg_jobs_per_claim=2,
        avg_labor_codes_per_job=3,
        optional_skip_rate=0.3
    )
    
    print(f"Generated {len(df)} rows of data")
    print(f"\nFirst few rows:")
    print(df.head().to_string())
    
    # Validate
    print("\n=== Validating Data ===")
    validation = loader.validate_data(df)
    print(f"Valid: {validation['is_valid']}")
    print(f"Summary: {validation['summary']}")
    
    # Add derived features
    print("\n=== Adding Derived Features ===")
    df_enhanced = loader.add_derived_features(df)
    print(f"Enhanced columns: {df_enhanced.columns.tolist()}")
    
    # Split by date
    print("\n=== Splitting by Date ===")
    splits = loader.split_by_date(df, train_end_date='2024-09-30')
    print(f"Train: {len(splits['train'])} rows")
    print(f"Val/Test: {len(splits['val'])} rows")
    
    # Export
    output_path = './minimal_warranty_ml_results/sample_data.csv'
    loader.export_for_ml(df_enhanced, output_path)
    
    # Summary
    print("\n=== Data Summary ===")
    summary = loader.get_data_summary(df)
    print(f"Shape: {summary['shape']}")
    print(f"Columns: {len(summary['columns'])}")
    print("\nCategorical summary (top campaigns):")
    print(summary['categorical_summary']['campaign_code'])


if __name__ == "__main__":
    main()
