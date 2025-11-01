"""
Complete Warranty Operation Predictor
Single unified system with tiered prediction and SHAP analysis

Usage:
    python warranty_predictor.py --data campaign_operation_performed.csv
    python warranty_predictor.py --data ops.csv --vehicle vehicles.csv
    python warranty_predictor.py --data ops.csv --vehicle vehicles.csv --dealer dealers.csv
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not installed. Run: pip install shap")
    print("   (Analysis will continue without SHAP feature importance)\n")

sns.set_style("whitegrid")


class WarrantyOperationPredictor:
    """
    Predicts whether optional warranty operations will be needed
    Adapts to available data automatically (tiered approach)
    """

    def __init__(self, output_dir='./results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.features = []
        self.label_encoders = {}
        self.training_results = {}
        self.available_tier = 1

    def load_data(self, operations_file, vehicle_file=None, dealer_file=None):
        """Load all data sources"""

        print("="*70)
        print("LOADING DATA")
        print("="*70)

        # Load operations (required)
        print(f"\n📥 Loading operations: {operations_file}")
        df = pd.read_csv(operations_file)

        required = ['CLAIM_JOB_ID', 'SCC_CODE', 'LABOR_CODE', 'OPTIONAL_FLAG', 'value']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        print(f"✓ Loaded {len(df):,} operations")
        print(f"  - Campaigns: {df['SCC_CODE'].nunique()}")
        print(f"  - Labor codes: {df['LABOR_CODE'].nunique()}")
        print(f"  - Claim jobs: {df['CLAIM_JOB_ID'].nunique()}")
        print(f"  - Optional ops: {df['OPTIONAL_FLAG'].sum():,} ({df['OPTIONAL_FLAG'].mean():.1%})")
        print(f"  - Performed ops: {df['value'].sum():,} ({df['value'].mean():.1%})")

        # Extract claim_id
        df['claim_id'] = df['CLAIM_JOB_ID'].str.rsplit('-', n=1).str[0]

        # Load vehicle data (optional - Tier 2)
        if vehicle_file and Path(vehicle_file).exists():
            print(f"\n📥 Loading vehicle data: {vehicle_file}")
            vehicle_df = pd.read_csv(vehicle_file)

            merge_key = 'claim_id' if 'claim_id' in vehicle_df.columns else 'CLAIM_JOB_ID'
            df = df.merge(vehicle_df, on=merge_key, how='left', suffixes=('', '_vehicle'))

            if 'vehicle_make' in df.columns:
                self.available_tier = max(self.available_tier, 2)
                print(f"✓ Merged vehicle data")
                print(f"  - Makes: {df['vehicle_make'].nunique()}")
                print(f"  - Models: {df['vehicle_model'].nunique() if 'vehicle_model' in df.columns else 'N/A'}")

                if 'mileage' in df.columns:
                    self.available_tier = max(self.available_tier, 3)
                    print(f"  - Mileage range: {df['mileage'].min():.0f} - {df['mileage'].max():.0f}")

        # Load dealer data (optional - Tier 4)
        if dealer_file and Path(dealer_file).exists():
            print(f"\n📥 Loading dealer data: {dealer_file}")
            dealer_df = pd.read_csv(dealer_file)

            merge_key = 'claim_id' if 'claim_id' in dealer_df.columns else 'CLAIM_JOB_ID'
            df = df.merge(dealer_df, on=merge_key, how='left', suffixes=('', '_dealer'))

            if 'dealer_id' in df.columns:
                self.available_tier = max(self.available_tier, 4)
                print(f"✓ Merged dealer data")
                print(f"  - Dealers: {df['dealer_id'].nunique()}")

        # Check for skipped operations
        optional_skipped = df[(df['OPTIONAL_FLAG'] == 1) & (df['value'] == 0)]
        print(f"\n📊 Data Quality Check:")
        print(f"  - Optional operations: {df['OPTIONAL_FLAG'].sum():,}")
        print(f"  - NOT performed: {len(optional_skipped):,} ({len(optional_skipped)/max(1, df['OPTIONAL_FLAG'].sum()):.1%})")

        if len(optional_skipped) == 0:
            print(f"\n  ⚠️  WARNING: No skipped optional operations found!")
            print(f"     Model will have limited predictive power.")

        print(f"\n✓ Data loaded - Using Tier {self.available_tier} features")

        return df

    def engineer_features(self, df):
        """Create all features based on available data"""

        print("\n" + "="*70)
        print("FEATURE ENGINEERING")
        print("="*70)

        df = df.copy()

        # Tier 1: Always available (baseline)
        print(f"\n🔧 Tier 1 Features (Baseline - Historical Patterns)...")

        df['campaign_encoded'] = self._encode(df, 'SCC_CODE', 'campaign')
        df['labor_code_encoded'] = self._encode(df, 'LABOR_CODE', 'labor')

        # Historical rates - THE KEY FEATURES
        df['campaign_rate'] = df.groupby('SCC_CODE')['value'].transform('mean')
        df['labor_rate'] = df.groupby('LABOR_CODE')['value'].transform('mean')
        df['campaign_labor_rate'] = df.groupby(['SCC_CODE', 'LABOR_CODE'])['value'].transform('mean')

        # Campaign complexity
        df['campaign_n_operations'] = df.groupby('SCC_CODE')['LABOR_CODE'].transform('nunique')
        df['campaign_n_optional'] = df.groupby('SCC_CODE')['OPTIONAL_FLAG'].transform('sum')
        df['campaign_optional_ratio'] = df['campaign_n_optional'] / df['campaign_n_operations']

        # Labor code frequency
        df['labor_frequency'] = df.groupby('LABOR_CODE')['CLAIM_JOB_ID'].transform('count')

        df['is_optional'] = df['OPTIONAL_FLAG'].astype(int)

        tier1_features = [
            'campaign_encoded', 'labor_code_encoded',
            'campaign_rate', 'labor_rate', 'campaign_labor_rate',
            'campaign_n_operations', 'campaign_optional_ratio',
            'labor_frequency', 'is_optional'
        ]

        print(f"  ✓ Created {len(tier1_features)} Tier 1 features")

        # Tier 2: Vehicle specifications (if available)
        tier2_features = tier1_features.copy()
        if self.available_tier >= 2 and 'vehicle_make' in df.columns:
            print(f"\n🔧 Tier 2 Features (Vehicle Specifications)...")

            df['make_encoded'] = self._encode(df, 'vehicle_make', 'make')
            df['model_encoded'] = self._encode(df, 'vehicle_model', 'model') if 'vehicle_model' in df.columns else 0
            df['vehicle_year'] = df['vehicle_year'].fillna(df['vehicle_year'].median()) if 'vehicle_year' in df.columns else 0

            # Vehicle-campaign interactions
            df['make_campaign_rate'] = df.groupby(['vehicle_make', 'SCC_CODE'])['value'].transform('mean')
            if 'vehicle_model' in df.columns:
                df['model_campaign_rate'] = df.groupby(['vehicle_model', 'SCC_CODE'])['value'].transform('mean')

            tier2_features.extend([
                'make_encoded', 'model_encoded', 'vehicle_year',
                'make_campaign_rate'
            ])
            if 'vehicle_model' in df.columns:
                tier2_features.append('model_campaign_rate')

            print(f"  ✓ Added {len(tier2_features) - len(tier1_features)} vehicle features")

        # Tier 3: Mileage/condition (if available)
        tier3_features = tier2_features.copy()
        if self.available_tier >= 3 and 'mileage' in df.columns:
            print(f"\n🔧 Tier 3 Features (Mileage & Condition)...")

            df['mileage'] = df['mileage'].fillna(df['mileage'].median())
            df['mileage_bracket'] = pd.cut(df['mileage'],
                bins=[0, 30000, 60000, 90000, 150000, np.inf],
                labels=[1, 2, 3, 4, 5]).astype(int)
            df['high_mileage'] = (df['mileage'] > 100000).astype(int)

            if 'claim_date' in df.columns and 'vehicle_year' in df.columns:
                df['claim_date'] = pd.to_datetime(df['claim_date'])
                df['vehicle_age'] = df['claim_date'].dt.year - df['vehicle_year']
            else:
                df['vehicle_age'] = 0

            tier3_features.extend(['mileage', 'mileage_bracket', 'high_mileage', 'vehicle_age'])

            print(f"  ✓ Added {len(tier3_features) - len(tier2_features)} mileage features")

        # Tier 4: Dealer/geographic (if available)
        tier4_features = tier3_features.copy()
        if self.available_tier >= 4 and 'dealer_id' in df.columns:
            print(f"\n🔧 Tier 4 Features (Dealer & Context)...")

            df['dealer_encoded'] = self._encode(df, 'dealer_id', 'dealer')
            df['dealer_rate'] = df.groupby('dealer_id')['value'].transform('mean')
            df['dealer_campaign_rate'] = df.groupby(['dealer_id', 'SCC_CODE'])['value'].transform('mean')

            if 'region' in df.columns:
                df['region_encoded'] = self._encode(df, 'region', 'region')
                tier4_features.append('region_encoded')

            if 'claim_date' in df.columns:
                df['claim_date'] = pd.to_datetime(df['claim_date'])
                df['claim_month'] = df['claim_date'].dt.month
                df['claim_quarter'] = df['claim_date'].dt.quarter
                tier4_features.extend(['claim_month', 'claim_quarter'])

            tier4_features.extend(['dealer_encoded', 'dealer_rate', 'dealer_campaign_rate'])

            print(f"  ✓ Added {len(tier4_features) - len(tier3_features)} dealer features")

        # Select features based on available tier
        if self.available_tier == 1:
            self.features = tier1_features
        elif self.available_tier == 2:
            self.features = tier2_features
        elif self.available_tier == 3:
            self.features = tier3_features
        else:
            self.features = tier4_features

        # Keep only features that exist
        self.features = [f for f in self.features if f in df.columns]

        print(f"\n✓ Total features for Tier {self.available_tier}: {len(self.features)}")

        return df

    def _encode(self, df, column, prefix):
        """Encode categorical variable"""
        if column not in df.columns:
            return 0

        key = f"{prefix}_encoder"
        if key not in self.label_encoders:
            encoder = LabelEncoder()
            valid = df[column].notna()
            encoder.fit(df.loc[valid, column].astype(str))
            self.label_encoders[key] = encoder

        encoder = self.label_encoders[key]
        result = np.full(len(df), -1, dtype=int)
        valid = df[column].notna()

        values = df.loc[valid, column].astype(str)
        known = values.isin(encoder.classes_)

        if known.any():
            result[valid.values & known.values] = encoder.transform(values[known])

        return result

    def train(self, df, target='value', test_size=0.2):
        """Train the model"""

        print("\n" + "="*70)
        print(f"TRAINING MODEL (Tier {self.available_tier})")
        print("="*70)

        # Prepare features
        X = df[self.features].fillna(0)
        y = df[target]

        print(f"\n📊 Dataset:")
        print(f"  - Samples: {len(X):,}")
        print(f"  - Features: {len(self.features)}")
        print(f"  - Positive class (performed): {y.sum():,} ({y.mean():.1%})")
        print(f"  - Negative class (skipped): {(~y.astype(bool)).sum():,} ({(1-y.mean()):.1%})")

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Train
        print(f"\n🎯 Training GradientBoosting model...")
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.model.fit(X_train, y_train)

        # Evaluate
        train_acc = self.model.score(X_train, y_train)
        test_acc = self.model.score(X_test, y_test)

        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        cm = confusion_matrix(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        print(f"\n📈 Performance:")
        print(f"  - Train Accuracy: {train_acc:.1%}")
        print(f"  - Test Accuracy:  {test_acc:.1%}")
        print(f"  - ROC-AUC:        {roc_auc:.3f}")

        print(f"\n📊 Confusion Matrix:")
        print(f"  TN: {cm[0,0]:5d}  |  FP: {cm[0,1]:5d}")
        print(f"  FN: {cm[1,0]:5d}  |  TP: {cm[1,1]:5d}")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\n🔝 Top 10 Features (Model Importance):")
        print(feature_importance.head(10).to_string(index=False))

        # Store results
        self.training_results = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'feature_importance': feature_importance,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

        # SHAP analysis
        if SHAP_AVAILABLE:
            print(f"\n🔍 Calculating SHAP values...")
            try:
                explainer = shap.TreeExplainer(self.model)
                sample_size = min(1000, len(X_test))
                X_shap = X_test.sample(n=sample_size, random_state=42)
                shap_values = explainer.shap_values(X_shap)

                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                elif len(shap_values.shape) > 2:
                    shap_values = shap_values[..., 1]

                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                shap_importance = pd.DataFrame({
                    'feature': self.features,
                    'shap_importance': mean_abs_shap
                }).sort_values('shap_importance', ascending=False)

                self.training_results['shap_values'] = shap_values
                self.training_results['shap_data'] = X_shap
                self.training_results['shap_importance'] = shap_importance

                print(f"\n🔝 Top 10 Features (SHAP Importance):")
                print(shap_importance.head(10).to_string(index=False))

            except Exception as e:
                print(f"⚠️  SHAP calculation failed: {e}")

        return self.training_results

    def create_visualizations(self):
        """Create all visualizations"""

        print("\n" + "="*70)
        print("CREATING VISUALIZATIONS")
        print("="*70)

        results = self.training_results

        # Main dashboard
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        fig.suptitle(f'Warranty Operation Predictor - Tier {self.available_tier} Analysis',
                    fontsize=16, fontweight='bold')

        # 1. Performance metrics
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')

        metrics_text = f"""
        MODEL PERFORMANCE (Tier {self.available_tier})

        Accuracy: {results['test_acc']:.1%}  |  ROC-AUC: {results['roc_auc']:.3f}  |  Features: {len(self.features)}

        True Positives: {results['confusion_matrix'][1,1]}  |  False Positives: {results['confusion_matrix'][0,1]}
        True Negatives: {results['confusion_matrix'][0,0]}  |  False Negatives: {results['confusion_matrix'][1,0]}
        """

        ax1.text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=12,
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        # 2. Feature importance (model-based)
        ax2 = fig.add_subplot(gs[1, 0])
        top_features = results['feature_importance'].head(12)
        ax2.barh(range(len(top_features)), top_features['importance'], alpha=0.7)
        ax2.set_yticks(range(len(top_features)))
        ax2.set_yticklabels(top_features['feature'], fontsize=9)
        ax2.set_xlabel('Importance')
        ax2.set_title('Top Features (Model)')
        ax2.invert_yaxis()
        ax2.grid(alpha=0.3, axis='x')

        # 3. Confusion matrix
        ax3 = fig.add_subplot(gs[1, 1])
        cm = results['confusion_matrix']
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', ax=ax3,
                   xticklabels=['Skip', 'Perform'],
                   yticklabels=['Skip', 'Perform'])
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        ax3.set_title('Confusion Matrix')

        # 4. ROC Curve
        ax4 = fig.add_subplot(gs[1, 2])
        fpr, tpr, _ = roc_curve(results['y_test'], results['y_pred_proba'])
        ax4.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {results["roc_auc"]:.3f})')
        ax4.plot([0, 1], [0, 1], 'k--', lw=2)
        ax4.set_xlabel('False Positive Rate')
        ax4.set_ylabel('True Positive Rate')
        ax4.set_title('ROC Curve')
        ax4.legend()
        ax4.grid(alpha=0.3)

        # 5. SHAP importance (if available)
        if 'shap_importance' in results:
            ax5 = fig.add_subplot(gs[2, 0])
            top_shap = results['shap_importance'].head(12)
            ax5.barh(range(len(top_shap)), top_shap['shap_importance'], alpha=0.7, color='coral')
            ax5.set_yticks(range(len(top_shap)))
            ax5.set_yticklabels(top_shap['feature'], fontsize=9)
            ax5.set_xlabel('Mean |SHAP value|')
            ax5.set_title('Top Features (SHAP)')
            ax5.invert_yaxis()
            ax5.grid(alpha=0.3, axis='x')

        # 6. Prediction distribution
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.hist([results['y_pred_proba'][results['y_test'] == 0],
                 results['y_pred_proba'][results['y_test'] == 1]],
                bins=30, label=['Actual: Skip', 'Actual: Perform'],
                alpha=0.7, edgecolor='black')
        ax6.set_xlabel('Predicted Probability')
        ax6.set_ylabel('Count')
        ax6.set_title('Prediction Distribution')
        ax6.legend()
        ax6.grid(alpha=0.3)

        # 7. Recommendations
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')

        if self.available_tier == 1:
            rec_text = "🎯 CURRENT: Tier 1 (Baseline)\n\n"
            rec_text += f"Accuracy: {results['test_acc']:.1%}\n\n"
            rec_text += "📊 TO IMPROVE:\n"
            rec_text += "• Collect vehicle data\n"
            rec_text += "  (make, model, year)\n"
            rec_text += "  Expected: +3-8% accuracy\n\n"
            rec_text += "• Add mileage data\n"
            rec_text += "  Expected: +2-5% accuracy"
        elif self.available_tier == 2:
            rec_text = "🎯 CURRENT: Tier 2\n"
            rec_text += "(Vehicle specs included)\n\n"
            rec_text += f"Accuracy: {results['test_acc']:.1%}\n\n"
            rec_text += "📊 TO IMPROVE:\n"
            rec_text += "• Add mileage data\n"
            rec_text += "  Expected: +2-5% accuracy"
        elif self.available_tier == 3:
            rec_text = "🎯 CURRENT: Tier 3\n"
            rec_text += "(Vehicle + mileage)\n\n"
            rec_text += f"Accuracy: {results['test_acc']:.1%}\n\n"
            rec_text += "✅ Model is strong!\n"
            rec_text += "Dealer data optional"
        else:
            rec_text = "🎯 CURRENT: Tier 4 (Max)\n\n"
            rec_text += f"Accuracy: {results['test_acc']:.1%}\n\n"
            rec_text += "✅ Using all available\n"
            rec_text += "   data sources!"

        ax7.text(0.1, 0.9, rec_text, ha='left', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

        plt.tight_layout()
        filepath = self.output_dir / 'analysis_dashboard.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved dashboard: {filepath}")
        plt.close()

        # SHAP detailed plot
        if 'shap_values' in results:
            self._plot_shap_details()

    def _plot_shap_details(self):
        """Create detailed SHAP visualization"""
        results = self.training_results
        shap_vals = results['shap_values']

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Bar plot
        shap_imp = results['shap_importance'].head(15)
        axes[0].barh(range(len(shap_imp)), shap_imp['shap_importance'], alpha=0.7)
        axes[0].set_yticks(range(len(shap_imp)))
        axes[0].set_yticklabels(shap_imp['feature'], fontsize=10)
        axes[0].set_xlabel('Mean |SHAP value|')
        axes[0].set_title(f'SHAP Feature Importance - Tier {self.available_tier}')
        axes[0].invert_yaxis()
        axes[0].grid(alpha=0.3, axis='x')

        # Beeswarm-style
        top_10 = shap_imp.head(10)
        for i, feature in enumerate(top_10['feature']):
            feat_idx = self.features.index(feature)
            values = shap_vals[:, feat_idx]

            sample_size = min(300, len(values))
            sample_idx = np.random.choice(len(values), sample_size, replace=False)

            y_pos = np.random.normal(i, 0.1, sample_size)
            axes[1].scatter(values[sample_idx], y_pos, alpha=0.3, s=10,
                          c=values[sample_idx], cmap='RdBu_r',
                          vmin=-np.abs(values).max(), vmax=np.abs(values).max())

        axes[1].set_yticks(range(10))
        axes[1].set_yticklabels(top_10['feature'], fontsize=10)
        axes[1].set_xlabel('SHAP value')
        axes[1].set_title('SHAP Value Distribution')
        axes[1].axvline(0, color='black', linestyle='--', alpha=0.5)
        axes[1].grid(alpha=0.3, axis='x')

        plt.tight_layout()
        filepath = self.output_dir / 'shap_analysis.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved SHAP analysis: {filepath}")
        plt.close()

    def save_model(self, filename='warranty_predictor.pkl'):
        """Save the trained model"""
        filepath = self.output_dir / filename

        save_data = {
            'model': self.model,
            'features': self.features,
            'label_encoders': self.label_encoders,
            'available_tier': self.available_tier
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"\n💾 Model saved: {filepath}")

    def load_model(self, filename='warranty_predictor.pkl'):
        """Load a saved model"""
        filepath = self.output_dir / filename

        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)

        self.model = save_data['model']
        self.features = save_data['features']
        self.label_encoders = save_data['label_encoders']
        self.available_tier = save_data['available_tier']

        print(f"✓ Model loaded: {filepath}")

    def predict(self, df):
        """Make predictions on new data"""
        X = df[self.features].fillna(0)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        confidence = 2 * np.abs(probabilities - 0.5)

        return predictions, probabilities, confidence


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Warranty Operation Predictor')
    parser.add_argument('--data', required=True, help='Operations CSV file')
    parser.add_argument('--vehicle', help='Optional: Vehicle data CSV')
    parser.add_argument('--dealer', help='Optional: Dealer data CSV')
    parser.add_argument('--output', default='./results', help='Output directory')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualizations')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("WARRANTY OPERATION PREDICTOR")
    print("="*70)

    # Initialize
    predictor = WarrantyOperationPredictor(output_dir=args.output)

    # Load data
    df = predictor.load_data(args.data, args.vehicle, args.dealer)

    # Engineer features
    df = predictor.engineer_features(df)

    # Train
    results = predictor.train(df)

    # Visualize
    if not args.no_viz:
        predictor.create_visualizations()

    # Save
    predictor.save_model()

    # Summary
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"\n✅ Tier {predictor.available_tier} model trained successfully")
    print(f"   Accuracy: {results['test_acc']:.1%}")
    print(f"   ROC-AUC: {results['roc_auc']:.3f}")
    print(f"\n📁 Results saved to: {args.output}/")
    print(f"   - analysis_dashboard.png")
    if 'shap_values' in results:
        print(f"   - shap_analysis.png")
    print(f"   - warranty_predictor.pkl")

    if predictor.available_tier == 1:
        print(f"\n💡 To improve: Add vehicle data with --vehicle flag")

    print("\n✨ Ready for predictions!")


if __name__ == "__main__":
    main()
