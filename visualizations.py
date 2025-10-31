"""
Visualization Module for Warranty Claim Analysis
Creates comprehensive visualizations at all hierarchy levels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class ClaimVisualizer:
    """Create visualizations for warranty claim analysis"""
    
    def __init__(self, output_dir: str = './minimal_warranty_ml_results/figures'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_skip_rate_distribution(self, analyzer, save: bool = True) -> None:
        """Plot distribution of skip rates across claims"""
        skip_rates = [claim.optional_skip_rate for claim in analyzer.claims 
                     if claim.optional_performed + claim.optional_skipped > 0]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(skip_rates, bins=30, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Optional Skip Rate')
        axes[0].set_ylabel('Number of Claims')
        axes[0].set_title('Distribution of Optional Labor Skip Rates')
        axes[0].axvline(np.mean(skip_rates), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(skip_rates):.2%}')
        axes[0].legend()
        
        # Box plot
        axes[1].boxplot(skip_rates, vert=True)
        axes[1].set_ylabel('Optional Skip Rate')
        axes[1].set_title('Skip Rate Distribution (Box Plot)')
        axes[1].set_xticklabels(['All Claims'])
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'skip_rate_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_campaign_analysis(self, analyzer, save: bool = True) -> None:
        """Analyze skip rates by campaign"""
        campaign_stats = []
        
        for claim in analyzer.claims:
            for job in claim.claim_jobs:
                optional_performed = job.optional_performed_count
                optional_skipped = job.optional_skipped_count
                total_optional = optional_performed + optional_skipped
                
                if total_optional > 0:
                    campaign_stats.append({
                        'campaign_code': job.campaign_code,
                        'skip_rate': optional_skipped / total_optional,
                        'total_cost': job.total_cost,
                        'labor_hours': job.total_labor_hours
                    })
        
        df = pd.DataFrame(campaign_stats)
        
        if df.empty:
            print("No data available for campaign analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Skip rate by campaign
        campaign_skip = df.groupby('campaign_code')['skip_rate'].agg(['mean', 'count'])
        campaign_skip = campaign_skip.sort_values('mean', ascending=False)
        
        axes[0, 0].barh(range(len(campaign_skip)), campaign_skip['mean'])
        axes[0, 0].set_yticks(range(len(campaign_skip)))
        axes[0, 0].set_yticklabels(campaign_skip.index)
        axes[0, 0].set_xlabel('Average Skip Rate')
        axes[0, 0].set_title('Average Skip Rate by Campaign')
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # Count by campaign
        axes[0, 1].bar(range(len(campaign_skip)), campaign_skip['count'])
        axes[0, 1].set_xticks(range(len(campaign_skip)))
        axes[0, 1].set_xticklabels(campaign_skip.index, rotation=45)
        axes[0, 1].set_ylabel('Number of Jobs')
        axes[0, 1].set_title('Job Count by Campaign')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Cost by campaign
        campaign_cost = df.groupby('campaign_code')['total_cost'].mean().sort_values(ascending=False)
        axes[1, 0].barh(range(len(campaign_cost)), campaign_cost)
        axes[1, 0].set_yticks(range(len(campaign_cost)))
        axes[1, 0].set_yticklabels(campaign_cost.index)
        axes[1, 0].set_xlabel('Average Total Cost ($)')
        axes[1, 0].set_title('Average Cost by Campaign')
        axes[1, 0].grid(axis='x', alpha=0.3)
        
        # Skip rate vs cost scatter
        axes[1, 1].scatter(df['total_cost'], df['skip_rate'], alpha=0.5)
        axes[1, 1].set_xlabel('Total Cost ($)')
        axes[1, 1].set_ylabel('Skip Rate')
        axes[1, 1].set_title('Skip Rate vs Total Cost')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'campaign_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_dealer_comparison(self, dealer_stats: pd.DataFrame, save: bool = True) -> None:
        """Compare dealer performance"""
        if dealer_stats.empty:
            print("No dealer statistics available")
            return
        
        # Sort by skip rate
        dealer_stats = dealer_stats.sort_values('optional_skip_rate', ascending=False).head(20)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Skip rate comparison
        axes[0, 0].barh(range(len(dealer_stats)), dealer_stats['optional_skip_rate'])
        axes[0, 0].set_yticks(range(len(dealer_stats)))
        axes[0, 0].set_yticklabels(dealer_stats['dealer_id'])
        axes[0, 0].set_xlabel('Optional Skip Rate')
        axes[0, 0].set_title('Top 20 Dealers by Skip Rate')
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # Claim count
        axes[0, 1].bar(range(len(dealer_stats)), dealer_stats['claim_count'])
        axes[0, 1].set_xticks(range(len(dealer_stats)))
        axes[0, 1].set_xticklabels(dealer_stats['dealer_id'], rotation=90)
        axes[0, 1].set_ylabel('Number of Claims')
        axes[0, 1].set_title('Claim Volume by Dealer')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Average cost
        axes[1, 0].bar(range(len(dealer_stats)), dealer_stats['avg_cost_per_claim'])
        axes[1, 0].set_xticks(range(len(dealer_stats)))
        axes[1, 0].set_xticklabels(dealer_stats['dealer_id'], rotation=90)
        axes[1, 0].set_ylabel('Average Cost per Claim ($)')
        axes[1, 0].set_title('Average Claim Cost by Dealer')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Skip rate vs complexity
        axes[1, 1].scatter(dealer_stats['avg_complexity'], dealer_stats['optional_skip_rate'], 
                          s=dealer_stats['claim_count']*2, alpha=0.6)
        axes[1, 1].set_xlabel('Average Claim Complexity (# Labor Codes)')
        axes[1, 1].set_ylabel('Optional Skip Rate')
        axes[1, 1].set_title('Skip Rate vs Complexity (size = claim volume)')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'dealer_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_temporal_trends(self, analyzer, save: bool = True) -> None:
        """Plot trends over time"""
        temporal_data = []
        
        for claim in analyzer.claims:
            temporal_data.append({
                'claim_date': claim.claim_date,
                'skip_rate': claim.optional_skip_rate,
                'total_cost': claim.total_cost,
                'campaign_count': claim.campaign_count,
                'complexity': claim.total_labor_codes
            })
        
        df = pd.DataFrame(temporal_data)
        df = df.sort_values('claim_date')
        
        # Aggregate by month
        df['year_month'] = df['claim_date'].dt.to_period('M')
        monthly = df.groupby('year_month').agg({
            'skip_rate': 'mean',
            'total_cost': 'mean',
            'campaign_count': 'mean',
            'complexity': 'mean'
        }).reset_index()
        monthly['year_month'] = monthly['year_month'].dt.to_timestamp()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Skip rate over time
        axes[0, 0].plot(monthly['year_month'], monthly['skip_rate'], marker='o', linewidth=2)
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('Average Skip Rate')
        axes[0, 0].set_title('Skip Rate Trend Over Time')
        axes[0, 0].grid(alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Cost over time
        axes[0, 1].plot(monthly['year_month'], monthly['total_cost'], marker='o', linewidth=2, color='green')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Average Total Cost ($)')
        axes[0, 1].set_title('Cost Trend Over Time')
        axes[0, 1].grid(alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Campaigns per claim over time
        axes[1, 0].plot(monthly['year_month'], monthly['campaign_count'], marker='o', linewidth=2, color='orange')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Average Campaigns per Claim')
        axes[1, 0].set_title('Claim Complexity (Campaigns) Over Time')
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Labor codes over time
        axes[1, 1].plot(monthly['year_month'], monthly['complexity'], marker='o', linewidth=2, color='red')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Average Labor Codes per Claim')
        axes[1, 1].set_title('Claim Complexity (Labor Codes) Over Time')
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'temporal_trends.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, importance_dict: Dict[str, pd.DataFrame], 
                               top_n: int = 15, save: bool = True) -> None:
        """Plot feature importance for all model levels"""
        n_models = len(importance_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 8))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (level, importance_df) in enumerate(importance_dict.items()):
            top_features = importance_df.head(top_n)
            
            axes[idx].barh(range(len(top_features)), top_features['importance'])
            axes[idx].set_yticks(range(len(top_features)))
            axes[idx].set_yticklabels(top_features['feature'])
            axes[idx].set_xlabel('Importance')
            axes[idx].set_title(f'{level.replace("_", " ").title()} - Top {top_n} Features')
            axes[idx].invert_yaxis()
            axes[idx].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrices(self, results_dict: Dict[str, Dict], save: bool = True) -> None:
        """Plot confusion matrices for all model levels"""
        n_models = len(results_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (level, results) in enumerate(results_dict.items()):
            cm = results['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['No Skip', 'Skip'],
                       yticklabels=['No Skip', 'Skip'])
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
            axes[idx].set_title(f'{level.replace("_", " ").title()}\nAccuracy: {results["test_score"]:.3f}')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_labor_code_analysis(self, skip_patterns: pd.DataFrame, 
                                top_n: int = 20, save: bool = True) -> None:
        """Analyze specific labor code skip patterns"""
        if skip_patterns.empty:
            print("No skip pattern data available")
            return
        
        # Get top skipped labor codes
        top_skipped = skip_patterns.head(top_n)
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 12))
        
        # Skip rate by labor code
        y_labels = [f"{row.name[1]} ({row.name[0]})" for _, row in top_skipped.iterrows()]
        
        axes[0].barh(range(len(top_skipped)), top_skipped['skip_rate'])
        axes[0].set_yticks(range(len(top_skipped)))
        axes[0].set_yticklabels(y_labels, fontsize=8)
        axes[0].set_xlabel('Skip Rate')
        axes[0].set_title(f'Top {top_n} Most Skipped Optional Labor Codes')
        axes[0].grid(axis='x', alpha=0.3)
        axes[0].invert_yaxis()
        
        # Frequency
        axes[1].barh(range(len(top_skipped)), top_skipped['total_occurrences'])
        axes[1].set_yticks(range(len(top_skipped)))
        axes[1].set_yticklabels(y_labels, fontsize=8)
        axes[1].set_xlabel('Total Occurrences')
        axes[1].set_title(f'Frequency of Top {top_n} Skipped Labor Codes')
        axes[1].grid(axis='x', alpha=0.3)
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'labor_code_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_executive_summary(self, analyzer, save: bool = True) -> None:
        """Create a single-page executive summary visualization"""
        summary_stats = analyzer.get_summary_statistics()
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Warranty Claims Analysis - Executive Summary', fontsize=16, fontweight='bold')
        
        # Key metrics
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        
        metrics_text = f"""
        Total Claims: {summary_stats['total_claims']:,}  |  Total Jobs: {summary_stats['total_jobs']:,}  |  Total Labor Codes: {summary_stats['total_labor_codes']:,}
        
        Average Campaigns per Claim: {summary_stats['avg_campaigns_per_claim']:.2f}  |  Overall Skip Rate: {summary_stats['overall_skip_rate']:.1%}  |  Optional Skip Rate: {summary_stats['overall_optional_skip_rate']:.1%}
        
        Total Cost: ${summary_stats['total_cost']:,.0f}  |  Average Cost per Claim: ${summary_stats['avg_cost_per_claim']:,.2f}
        """
        
        ax1.text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Skip rate distribution
        ax2 = fig.add_subplot(gs[1, 0])
        skip_rates = [claim.optional_skip_rate for claim in analyzer.claims 
                     if claim.optional_performed + claim.optional_skipped > 0]
        ax2.hist(skip_rates, bins=20, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Skip Rate')
        ax2.set_ylabel('Count')
        ax2.set_title('Skip Rate Distribution')
        
        # Cost distribution
        ax3 = fig.add_subplot(gs[1, 1])
        costs = [claim.total_cost for claim in analyzer.claims]
        ax3.hist(costs, bins=20, edgecolor='black', alpha=0.7, color='green')
        ax3.set_xlabel('Total Cost ($)')
        ax3.set_ylabel('Count')
        ax3.set_title('Cost Distribution')
        
        # Complexity distribution
        ax4 = fig.add_subplot(gs[1, 2])
        complexity = [claim.total_labor_codes for claim in analyzer.claims]
        ax4.hist(complexity, bins=20, edgecolor='black', alpha=0.7, color='orange')
        ax4.set_xlabel('Labor Codes')
        ax4.set_ylabel('Count')
        ax4.set_title('Complexity Distribution')
        
        # Campaign distribution
        ax5 = fig.add_subplot(gs[2, :])
        campaign_counts = {}
        for claim in analyzer.claims:
            for job in claim.claim_jobs:
                campaign_counts[job.campaign_code] = campaign_counts.get(job.campaign_code, 0) + 1
        
        campaigns = sorted(campaign_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ax5.bar([c[0] for c in campaigns], [c[1] for c in campaigns])
        ax5.set_xlabel('Campaign Code')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Top 10 Campaigns by Frequency')
        ax5.tick_params(axis='x', rotation=45)
        
        if save:
            plt.savefig(self.output_dir / 'executive_summary.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Example usage"""
    from claim_analyzer import ClaimAnalyzer
    from data_loader import ClaimDataLoader
    
    # Generate synthetic data
    loader = ClaimDataLoader()
    df = loader.create_synthetic_data(n_claims=200, optional_skip_rate=0.3)
    
    # Load into analyzer
    analyzer = ClaimAnalyzer()
    analyzer.load_from_dataframe(df)
    
    # Create visualizations
    viz = ClaimVisualizer()
    
    print("Creating visualizations...")
    
    viz.plot_skip_rate_distribution(analyzer)
    viz.plot_campaign_analysis(analyzer)
    
    dealer_stats = analyzer.analyze_dealer_patterns()
    viz.plot_dealer_comparison(dealer_stats)
    
    viz.plot_temporal_trends(analyzer)
    
    skip_patterns = analyzer.analyze_skip_patterns()
    viz.plot_labor_code_analysis(skip_patterns)
    
    viz.create_executive_summary(analyzer)
    
    print(f"\nAll visualizations saved to: {viz.output_dir}")


if __name__ == "__main__":
    main()
