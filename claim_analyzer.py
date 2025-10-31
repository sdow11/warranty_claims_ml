"""
Claim-Level Analysis Module
Analyzes warranty claims with full three-level hierarchy:
1. Claim (One Vehicle Visit)
2. Claim Job (Campaign/Job within claim)
3. Labor Code (Individual operations)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class LaborCode:
    """Individual labor operation"""
    code: str
    description: str
    performed: bool
    is_optional: bool
    labor_hours: float = 0.0
    labor_cost: float = 0.0
    parts_cost: float = 0.0


@dataclass
class ClaimJob:
    """Campaign/Job within a claim"""
    job_id: str
    campaign_code: str
    labor_codes: List[LaborCode]
    total_labor_hours: float = 0.0
    total_labor_cost: float = 0.0
    total_parts_cost: float = 0.0
    
    @property
    def total_cost(self) -> float:
        return self.total_labor_cost + self.total_parts_cost
    
    @property
    def performed_count(self) -> int:
        return sum(1 for lc in self.labor_codes if lc.performed)
    
    @property
    def skipped_count(self) -> int:
        return sum(1 for lc in self.labor_codes if not lc.performed)
    
    @property
    def optional_performed_count(self) -> int:
        return sum(1 for lc in self.labor_codes if lc.is_optional and lc.performed)
    
    @property
    def optional_skipped_count(self) -> int:
        return sum(1 for lc in self.labor_codes if lc.is_optional and not lc.performed)


@dataclass
class Claim:
    """Complete claim for one vehicle visit"""
    claim_id: str
    vehicle_id: str
    claim_date: pd.Timestamp
    dealer_id: str
    claim_jobs: List[ClaimJob]
    
    # Vehicle attributes
    vehicle_make: Optional[str] = None
    vehicle_model: Optional[str] = None
    vehicle_year: Optional[int] = None
    mileage: Optional[float] = None
    
    @property
    def total_cost(self) -> float:
        return sum(job.total_cost for job in self.claim_jobs)
    
    @property
    def total_labor_hours(self) -> float:
        return sum(job.total_labor_hours for job in self.claim_jobs)
    
    @property
    def campaign_count(self) -> int:
        return len(self.claim_jobs)
    
    @property
    def total_labor_codes(self) -> int:
        return sum(len(job.labor_codes) for job in self.claim_jobs)
    
    @property
    def total_performed(self) -> int:
        return sum(job.performed_count for job in self.claim_jobs)
    
    @property
    def total_skipped(self) -> int:
        return sum(job.skipped_count for job in self.claim_jobs)
    
    @property
    def optional_performed(self) -> int:
        return sum(job.optional_performed_count for job in self.claim_jobs)
    
    @property
    def optional_skipped(self) -> int:
        return sum(job.optional_skipped_count for job in self.claim_jobs)
    
    @property
    def skip_rate(self) -> float:
        total = self.total_labor_codes
        return self.total_skipped / total if total > 0 else 0.0
    
    @property
    def optional_skip_rate(self) -> float:
        total_optional = self.optional_performed + self.optional_skipped
        return self.optional_skipped / total_optional if total_optional > 0 else 0.0


class ClaimAnalyzer:
    """Comprehensive claim-level analysis"""
    
    def __init__(self):
        self.claims: List[Claim] = []
        
    def load_claims(self, claims_data: List[Dict]) -> None:
        """Load claim data from structured format"""
        for claim_dict in claims_data:
            claim_jobs = []
            
            for job_dict in claim_dict.get('claim_jobs', []):
                labor_codes = []
                
                for lc_dict in job_dict.get('labor_codes', []):
                    labor_code = LaborCode(
                        code=lc_dict['code'],
                        description=lc_dict.get('description', ''),
                        performed=lc_dict['performed'],
                        is_optional=lc_dict.get('is_optional', True),
                        labor_hours=lc_dict.get('labor_hours', 0.0),
                        labor_cost=lc_dict.get('labor_cost', 0.0),
                        parts_cost=lc_dict.get('parts_cost', 0.0)
                    )
                    labor_codes.append(labor_code)
                
                claim_job = ClaimJob(
                    job_id=job_dict['job_id'],
                    campaign_code=job_dict['campaign_code'],
                    labor_codes=labor_codes,
                    total_labor_hours=job_dict.get('total_labor_hours', 0.0),
                    total_labor_cost=job_dict.get('total_labor_cost', 0.0),
                    total_parts_cost=job_dict.get('total_parts_cost', 0.0)
                )
                claim_jobs.append(claim_job)
            
            claim = Claim(
                claim_id=claim_dict['claim_id'],
                vehicle_id=claim_dict['vehicle_id'],
                claim_date=pd.to_datetime(claim_dict['claim_date']),
                dealer_id=claim_dict['dealer_id'],
                claim_jobs=claim_jobs,
                vehicle_make=claim_dict.get('vehicle_make'),
                vehicle_model=claim_dict.get('vehicle_model'),
                vehicle_year=claim_dict.get('vehicle_year'),
                mileage=claim_dict.get('mileage')
            )
            self.claims.append(claim)
    
    def load_from_dataframe(self, df: pd.DataFrame) -> None:
        """
        Load claims from flat dataframe with columns:
        - claim_id, vehicle_id, claim_date, dealer_id
        - job_id, campaign_code
        - labor_code, labor_description, performed, is_optional
        - labor_hours, labor_cost, parts_cost
        """
        claims_dict = defaultdict(lambda: {
            'claim_jobs': defaultdict(lambda: {'labor_codes': []})
        })
        
        for _, row in df.iterrows():
            claim_id = row['claim_id']
            job_id = row['job_id']
            
            # Initialize claim info if first time
            if 'claim_id' not in claims_dict[claim_id]:
                claims_dict[claim_id].update({
                    'claim_id': claim_id,
                    'vehicle_id': row['vehicle_id'],
                    'claim_date': row['claim_date'],
                    'dealer_id': row['dealer_id'],
                    'vehicle_make': row.get('vehicle_make'),
                    'vehicle_model': row.get('vehicle_model'),
                    'vehicle_year': row.get('vehicle_year'),
                    'mileage': row.get('mileage')
                })
            
            # Initialize job info if first time
            if 'job_id' not in claims_dict[claim_id]['claim_jobs'][job_id]:
                claims_dict[claim_id]['claim_jobs'][job_id].update({
                    'job_id': job_id,
                    'campaign_code': row['campaign_code'],
                    'total_labor_hours': 0.0,
                    'total_labor_cost': 0.0,
                    'total_parts_cost': 0.0
                })
            
            # Add labor code
            labor_code = {
                'code': row['labor_code'],
                'description': row.get('labor_description', ''),
                'performed': row['performed'],
                'is_optional': row.get('is_optional', True),
                'labor_hours': row.get('labor_hours', 0.0),
                'labor_cost': row.get('labor_cost', 0.0),
                'parts_cost': row.get('parts_cost', 0.0)
            }
            claims_dict[claim_id]['claim_jobs'][job_id]['labor_codes'].append(labor_code)
            
            # Accumulate job totals
            if row['performed']:
                claims_dict[claim_id]['claim_jobs'][job_id]['total_labor_hours'] += row.get('labor_hours', 0.0)
                claims_dict[claim_id]['claim_jobs'][job_id]['total_labor_cost'] += row.get('labor_cost', 0.0)
                claims_dict[claim_id]['claim_jobs'][job_id]['total_parts_cost'] += row.get('parts_cost', 0.0)
        
        # Convert to list format
        claims_data = []
        for claim_id, claim_info in claims_dict.items():
            claim_info['claim_jobs'] = list(claim_info['claim_jobs'].values())
            claims_data.append(claim_info)
        
        self.load_claims(claims_data)
    
    def get_claim_features(self, claim: Claim) -> Dict[str, any]:
        """Extract comprehensive features for a claim"""
        features = {
            # Basic identifiers
            'claim_id': claim.claim_id,
            'vehicle_id': claim.vehicle_id,
            'dealer_id': claim.dealer_id,
            'claim_date': claim.claim_date,
            
            # Vehicle attributes
            'vehicle_make': claim.vehicle_make,
            'vehicle_model': claim.vehicle_model,
            'vehicle_year': claim.vehicle_year,
            'mileage': claim.mileage,
            
            # Aggregate statistics
            'campaign_count': claim.campaign_count,
            'total_labor_codes': claim.total_labor_codes,
            'total_performed': claim.total_performed,
            'total_skipped': claim.total_skipped,
            'optional_performed': claim.optional_performed,
            'optional_skipped': claim.optional_skipped,
            
            # Rates
            'skip_rate': claim.skip_rate,
            'optional_skip_rate': claim.optional_skip_rate,
            
            # Financial
            'total_cost': claim.total_cost,
            'total_labor_hours': claim.total_labor_hours,
            'avg_cost_per_job': claim.total_cost / claim.campaign_count if claim.campaign_count > 0 else 0,
            
            # Complexity indicators
            'has_multiple_campaigns': claim.campaign_count > 1,
            'high_complexity': claim.total_labor_codes > 10,
            'mixed_performance': claim.total_performed > 0 and claim.total_skipped > 0
        }
        
        # Campaign-specific patterns
        campaigns = [job.campaign_code for job in claim.claim_jobs]
        features['unique_campaigns'] = len(set(campaigns))
        features['campaign_codes'] = ','.join(sorted(set(campaigns)))
        
        # Per-job statistics
        if claim.claim_jobs:
            features['avg_labor_codes_per_job'] = claim.total_labor_codes / claim.campaign_count
            features['max_job_cost'] = max(job.total_cost for job in claim.claim_jobs)
            features['min_job_cost'] = min(job.total_cost for job in claim.claim_jobs)
            features['cost_variance'] = np.var([job.total_cost for job in claim.claim_jobs])
        
        return features
    
    def create_features_dataframe(self) -> pd.DataFrame:
        """Create dataframe with all claim-level features"""
        features_list = [self.get_claim_features(claim) for claim in self.claims]
        return pd.DataFrame(features_list)
    
    def analyze_skip_patterns(self) -> pd.DataFrame:
        """Analyze patterns in optional labor code skipping"""
        patterns = []
        
        for claim in self.claims:
            for job in claim.claim_jobs:
                for labor_code in job.labor_codes:
                    if labor_code.is_optional:
                        patterns.append({
                            'claim_id': claim.claim_id,
                            'campaign_code': job.campaign_code,
                            'labor_code': labor_code.code,
                            'labor_description': labor_code.description,
                            'performed': labor_code.performed,
                            'dealer_id': claim.dealer_id,
                            'vehicle_make': claim.vehicle_make,
                            'vehicle_model': claim.vehicle_model,
                            'mileage': claim.mileage,
                            'claim_total_cost': claim.total_cost,
                            'claim_complexity': claim.total_labor_codes
                        })
        
        df = pd.DataFrame(patterns)
        
        # Aggregate statistics
        if not df.empty:
            summary = df.groupby(['campaign_code', 'labor_code', 'labor_description']).agg({
                'performed': ['sum', 'count', 'mean'],
                'claim_id': 'nunique'
            }).round(3)
            summary.columns = ['performed_count', 'total_occurrences', 'performance_rate', 'unique_claims']
            summary['skip_rate'] = 1 - summary['performance_rate']
            summary = summary.sort_values('skip_rate', ascending=False)
            
            return summary
        
        return pd.DataFrame()
    
    def analyze_dealer_patterns(self) -> pd.DataFrame:
        """Analyze dealer-specific patterns"""
        dealer_stats = []
        
        for dealer_id in set(claim.dealer_id for claim in self.claims):
            dealer_claims = [c for c in self.claims if c.dealer_id == dealer_id]
            
            total_optional = sum(c.optional_performed + c.optional_skipped for c in dealer_claims)
            total_optional_skipped = sum(c.optional_skipped for c in dealer_claims)
            
            dealer_stats.append({
                'dealer_id': dealer_id,
                'claim_count': len(dealer_claims),
                'avg_cost_per_claim': np.mean([c.total_cost for c in dealer_claims]),
                'avg_labor_hours': np.mean([c.total_labor_hours for c in dealer_claims]),
                'avg_campaigns_per_claim': np.mean([c.campaign_count for c in dealer_claims]),
                'optional_skip_rate': total_optional_skipped / total_optional if total_optional > 0 else 0,
                'avg_complexity': np.mean([c.total_labor_codes for c in dealer_claims])
            })
        
        df = pd.DataFrame(dealer_stats)
        return df.sort_values('optional_skip_rate', ascending=False)
    
    def analyze_campaign_combinations(self) -> pd.DataFrame:
        """Analyze how campaigns appear together"""
        combinations = defaultdict(int)
        
        for claim in self.claims:
            if claim.campaign_count > 1:
                campaigns = tuple(sorted(set(job.campaign_code for job in claim.claim_jobs)))
                combinations[campaigns] += 1
        
        results = [
            {'campaign_combination': ' + '.join(combo), 'frequency': count}
            for combo, count in combinations.items()
        ]
        
        df = pd.DataFrame(results)
        return df.sort_values('frequency', ascending=False) if not df.empty else df
    
    def get_summary_statistics(self) -> Dict[str, any]:
        """Get overall summary statistics"""
        if not self.claims:
            return {}
        
        return {
            'total_claims': len(self.claims),
            'total_jobs': sum(c.campaign_count for c in self.claims),
            'total_labor_codes': sum(c.total_labor_codes for c in self.claims),
            'avg_campaigns_per_claim': np.mean([c.campaign_count for c in self.claims]),
            'avg_labor_codes_per_claim': np.mean([c.total_labor_codes for c in self.claims]),
            'overall_skip_rate': np.mean([c.skip_rate for c in self.claims]),
            'overall_optional_skip_rate': np.mean([c.optional_skip_rate for c in self.claims if c.optional_performed + c.optional_skipped > 0]),
            'total_cost': sum(c.total_cost for c in self.claims),
            'avg_cost_per_claim': np.mean([c.total_cost for c in self.claims]),
            'unique_vehicles': len(set(c.vehicle_id for c in self.claims)),
            'unique_dealers': len(set(c.dealer_id for c in self.claims)),
            'unique_campaigns': len(set(job.campaign_code for claim in self.claims for job in claim.claim_jobs))
        }


def main():
    """Example usage"""
    # Example data structure
    example_data = [
        {
            'claim_id': 'CLM001',
            'vehicle_id': 'VIN12345',
            'claim_date': '2024-01-15',
            'dealer_id': 'DLR001',
            'vehicle_make': 'Toyota',
            'vehicle_model': 'Camry',
            'vehicle_year': 2020,
            'mileage': 35000,
            'claim_jobs': [
                {
                    'job_id': 'JOB001',
                    'campaign_code': 'S3494',
                    'total_labor_hours': 2.5,
                    'total_labor_cost': 250.0,
                    'total_parts_cost': 150.0,
                    'labor_codes': [
                        {
                            'code': '2589700',
                            'description': 'Replace component A',
                            'performed': True,
                            'is_optional': False,
                            'labor_hours': 1.5,
                            'labor_cost': 150.0,
                            'parts_cost': 100.0
                        },
                        {
                            'code': '2557300',
                            'description': 'Inspect component B',
                            'performed': True,
                            'is_optional': True,
                            'labor_hours': 1.0,
                            'labor_cost': 100.0,
                            'parts_cost': 50.0
                        },
                        {
                            'code': '1702800',
                            'description': 'Optional alignment',
                            'performed': False,
                            'is_optional': True,
                            'labor_hours': 0.5,
                            'labor_cost': 50.0,
                            'parts_cost': 0.0
                        }
                    ]
                },
                {
                    'job_id': 'JOB002',
                    'campaign_code': 'S3757',
                    'total_labor_hours': 1.0,
                    'total_labor_cost': 100.0,
                    'total_parts_cost': 75.0,
                    'labor_codes': [
                        {
                            'code': '2851300',
                            'description': 'Software update',
                            'performed': True,
                            'is_optional': True,
                            'labor_hours': 1.0,
                            'labor_cost': 100.0,
                            'parts_cost': 75.0
                        }
                    ]
                }
            ]
        }
    ]
    
    analyzer = ClaimAnalyzer()
    analyzer.load_claims(example_data)
    
    print("=== Claim Summary ===")
    summary = analyzer.get_summary_statistics()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\n=== Claim Features ===")
    features_df = analyzer.create_features_dataframe()
    print(features_df.to_string())
    
    print("\n=== Skip Patterns ===")
    skip_patterns = analyzer.analyze_skip_patterns()
    print(skip_patterns.to_string())


if __name__ == "__main__":
    main()
