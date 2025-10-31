"""
Complete Warranty Claims Analysis Workflow
Orchestrates all modules for end-to-end analysis
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

from claim_analyzer import ClaimAnalyzer
from predictive_models import ClaimLevelPredictor, EnsemblePredictor
from data_loader import ClaimDataLoader
from visualizations import ClaimVisualizer


def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def run_complete_analysis(data_source='synthetic', 
                         n_claims=500,
                         create_visualizations=True,
                         train_models=True):
    """
    Run complete warranty claims analysis workflow
    
    Args:
        data_source: 'synthetic', 'csv', or 'json'
        n_claims: Number of claims if generating synthetic data
        create_visualizations: Whether to create plots
        train_models: Whether to train predictive models
    """
    
    print_section("STEP 1: DATA LOADING")
    
    loader = ClaimDataLoader()
    
    if data_source == 'synthetic':
        print(f"Generating {n_claims} synthetic claims...")
        df = loader.create_synthetic_data(
            n_claims=n_claims,
            avg_jobs_per_claim=2,
            avg_labor_codes_per_job=3,
            optional_skip_rate=0.3,
            seed=42
        )
        print(f"✓ Generated {len(df)} rows of data")
    elif data_source == 'csv':
        print("Loading from CSV...")
        df = loader.load_from_csv('warranty_claims.csv')
        print(f"✓ Loaded {len(df)} rows")
    elif data_source == 'json':
        print("Loading from JSON...")
        claims_data = loader.load_from_json('warranty_claims.json')
        print(f"✓ Loaded {len(claims_data)} claims")
        # Convert to DataFrame for consistent processing
        # This would need to be implemented based on JSON structure
    else:
        raise ValueError(f"Unknown data source: {data_source}")
    
    print(f"\nData shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head(3).to_string())
    
    # Validate data
    print_section("STEP 2: DATA VALIDATION")
    validation = loader.validate_data(df)
    
    print(f"Valid: {validation['is_valid']}")
    if validation['missing_columns']:
        print(f"Missing columns: {validation['missing_columns']}")
    if validation['warnings']:
        print("Warnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    
    print("\nSummary:")
    for key, value in validation['summary'].items():
        print(f"  {key}: {value}")
    
    # Add derived features
    print_section("STEP 3: FEATURE ENGINEERING")
    df_enhanced = loader.add_derived_features(df)
    new_cols = set(df_enhanced.columns) - set(df.columns)
    print(f"Added {len(new_cols)} derived features:")
    for col in sorted(new_cols):
        print(f"  - {col}")
    
    # Save processed data
    output_path = Path('./minimal_warranty_ml_results/processed_data.csv')
    loader.export_for_ml(df_enhanced, output_path)
    
    # Load into analyzer
    print_section("STEP 4: CLAIM-LEVEL ANALYSIS")
    analyzer = ClaimAnalyzer()
    analyzer.load_from_dataframe(df_enhanced)
    print(f"✓ Loaded {len(analyzer.claims)} claims")
    
    # Summary statistics
    summary = analyzer.get_summary_statistics()
    print("\nSummary Statistics:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Skip patterns
    print_section("STEP 5: SKIP PATTERN ANALYSIS")
    skip_patterns = analyzer.analyze_skip_patterns()
    
    if not skip_patterns.empty:
        print("Top 10 Most Skipped Optional Labor Codes:")
        print(skip_patterns.head(10).to_string())
        
        print(f"\nTotal unique labor codes analyzed: {len(skip_patterns)}")
        print(f"Average skip rate: {skip_patterns['skip_rate'].mean():.2%}")
        print(f"Max skip rate: {skip_patterns['skip_rate'].max():.2%}")
    else:
        print("No skip patterns found in data")
    
    # Dealer patterns
    print_section("STEP 6: DEALER ANALYSIS")
    dealer_patterns = analyzer.analyze_dealer_patterns()
    
    if not dealer_patterns.empty:
        print("Top 10 Dealers by Skip Rate:")
        print(dealer_patterns.head(10).to_string())
        
        print(f"\nTotal dealers: {len(dealer_patterns)}")
        print(f"Average dealer skip rate: {dealer_patterns['optional_skip_rate'].mean():.2%}")
        print(f"Highest dealer skip rate: {dealer_patterns['optional_skip_rate'].max():.2%}")
    else:
        print("No dealer patterns found")
    
    # Campaign combinations
    print_section("STEP 7: CAMPAIGN COMBINATION ANALYSIS")
    campaign_combos = analyzer.analyze_campaign_combinations()
    
    if not campaign_combos.empty:
        print("Top 10 Campaign Combinations:")
        print(campaign_combos.head(10).to_string())
    else:
        print("No multi-campaign claims found")
    
    # Prepare features (needed for return even if not training models)
    features_df = analyzer.create_features_dataframe()
    
    # Predictive modeling
    if train_models:
        print_section("STEP 8: PREDICTIVE MODELING")
        
        predictor = ClaimLevelPredictor('random_forest')
        
        # Claim-level model
        print("\n>>> Training Claim-Level Model <<<")
        X_claim, y_claim = predictor.prepare_claim_level_data(features_df)
        print(f"Dataset: {len(X_claim)} samples, {X_claim.shape[1]} features")
        print(f"Target distribution: {y_claim.value_counts().to_dict()}")
        
        results_claim = predictor.train_claim_model(X_claim, y_claim)
        print(f"\nTrain Score: {results_claim['train_score']:.3f}")
        print(f"Test Score: {results_claim['test_score']:.3f}")
        print(f"ROC-AUC: {results_claim['roc_auc']:.3f}")
        print("\nClassification Report:")
        print(results_claim['classification_report'])
        
        # Job-level model
        print("\n>>> Training Job-Level Model <<<")
        X_job, y_job = predictor.prepare_job_level_data(analyzer)
        print(f"Dataset: {len(X_job)} samples, {X_job.shape[1]} features")
        print(f"Target distribution: {y_job.value_counts().to_dict()}")
        
        results_job = predictor.train_job_model(X_job, y_job)
        print(f"\nTrain Score: {results_job['train_score']:.3f}")
        print(f"Test Score: {results_job['test_score']:.3f}")
        print(f"ROC-AUC: {results_job['roc_auc']:.3f}")
        
        # Labor code-level model
        print("\n>>> Training Labor Code-Level Model <<<")
        X_labor, y_labor = predictor.prepare_labor_code_level_data(analyzer)
        print(f"Dataset: {len(X_labor)} samples, {X_labor.shape[1]} features")
        print(f"Target distribution: {y_labor.value_counts().to_dict()}")
        
        results_labor = predictor.train_labor_code_model(X_labor, y_labor)
        print(f"\nTrain Score: {results_labor['train_score']:.3f}")
        print(f"Test Score: {results_labor['test_score']:.3f}")
        print(f"ROC-AUC: {results_labor['roc_auc']:.3f}")
        
        # Feature importance
        print_section("STEP 9: FEATURE IMPORTANCE")
        
        for level, importance_df in predictor.get_feature_importance().items():
            print(f"\n>>> {level.upper()} <<<")
            print("Top 10 Features:")
            print(importance_df.head(10).to_string(index=False))
    
    # Visualizations
    if create_visualizations:
        print_section("STEP 10: CREATING VISUALIZATIONS")
        
        viz = ClaimVisualizer()
        
        print("Creating skip rate distribution...")
        viz.plot_skip_rate_distribution(analyzer, save=True)
        
        print("Creating campaign analysis...")
        viz.plot_campaign_analysis(analyzer, save=True)
        
        if not dealer_patterns.empty:
            print("Creating dealer comparison...")
            viz.plot_dealer_comparison(dealer_patterns, save=True)
        
        print("Creating temporal trends...")
        viz.plot_temporal_trends(analyzer, save=True)
        
        if not skip_patterns.empty:
            print("Creating labor code analysis...")
            viz.plot_labor_code_analysis(skip_patterns, save=True)
        
        print("Creating executive summary...")
        viz.create_executive_summary(analyzer, save=True)
        
        if train_models:
            print("Creating feature importance plots...")
            viz.plot_feature_importance(predictor.get_feature_importance(), save=True)
            
            print("Creating confusion matrices...")
            results_dict = {
                'claim': results_claim,
                'job': results_job,
                'labor_code': results_labor
            }
            viz.plot_confusion_matrices(results_dict, save=True)
        
        print(f"\n✓ All visualizations saved to: {viz.output_dir}")
    
    # Final summary
    print_section("ANALYSIS COMPLETE")
    
    print("Key Findings:")
    print(f"  • Analyzed {summary['total_claims']} claims across {summary['unique_vehicles']} vehicles")
    print(f"  • Overall optional skip rate: {summary['overall_optional_skip_rate']:.1%}")
    print(f"  • Total warranty cost: ${summary['total_cost']:,.0f}")
    print(f"  • Average cost per claim: ${summary['avg_cost_per_claim']:,.2f}")
    
    if train_models:
        print(f"\n  • Claim-level model ROC-AUC: {results_claim['roc_auc']:.3f}")
        print(f"  • Job-level model ROC-AUC: {results_job['roc_auc']:.3f}")
        print(f"  • Labor-code-level model ROC-AUC: {results_labor['roc_auc']:.3f}")
    
    if not skip_patterns.empty:
        most_skipped = skip_patterns.iloc[0]
        print(f"\n  • Most skipped labor code: {most_skipped.name[1]} ({most_skipped.name[0]})")
        print(f"    Skip rate: {most_skipped['skip_rate']:.1%}, Occurrences: {int(most_skipped['total_occurrences'])}")
    
    if not dealer_patterns.empty:
        worst_dealer = dealer_patterns.iloc[0]
        print(f"\n  • Highest skip rate dealer: {worst_dealer['dealer_id']}")
        print(f"    Skip rate: {worst_dealer['optional_skip_rate']:.1%}, Claims: {int(worst_dealer['claim_count'])}")
    
    print("\nOutput files:")
    print(f"  • Processed data: {output_path}")
    if create_visualizations:
        print(f"  • Visualizations: {viz.output_dir}")
    
    return {
        'analyzer': analyzer,
        'predictor': predictor if train_models else None,
        'features_df': features_df,
        'summary': summary
    }


def run_quick_analysis(n_claims=100):
    """Quick analysis with minimal visualizations"""
    print("Running quick analysis (no visualizations, no models)...")
    return run_complete_analysis(
        data_source='synthetic',
        n_claims=n_claims,
        create_visualizations=False,
        train_models=False
    )


def run_full_analysis(n_claims=500):
    """Full analysis with all features"""
    print("Running full analysis (with visualizations and models)...")
    return run_complete_analysis(
        data_source='synthetic',
        n_claims=n_claims,
        create_visualizations=True,
        train_models=True
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run warranty claims analysis')
    parser.add_argument('--mode', choices=['quick', 'full'], default='full',
                       help='Analysis mode: quick (fast) or full (comprehensive)')
    parser.add_argument('--claims', type=int, default=500,
                       help='Number of claims to generate (for synthetic data)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--no-models', action='store_true',
                       help='Skip model training')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        results = run_quick_analysis(n_claims=args.claims)
    else:
        results = run_complete_analysis(
            data_source='synthetic',
            n_claims=args.claims,
            create_visualizations=not args.no_viz,
            train_models=not args.no_models
        )
    
    print("\n✓ Analysis pipeline completed successfully!")
