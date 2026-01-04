"""
NBA Player Performance Prediction - ML Model Training
This script trains multiple models to predict points, rebounds, and assists
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class NBAPredictor:
    """
    Train and evaluate models for NBA player performance prediction
    """
    
    def __init__(self, features_csv):
        """
        Load the feature-engineered data
        """
        self.report_lines = []
        self.report_lines.append("="*70)
        self.report_lines.append("NBA PLAYER PERFORMANCE PREDICTION")
        self.report_lines.append("="*70)
        
        self.df = pd.read_csv(features_csv)
        self.df['GAME_DATE'] = pd.to_datetime(self.df['GAME_DATE'])
        self.df = self.df.sort_values('GAME_DATE').reset_index(drop=True)
        
        self.report_lines.append(f"\nLoaded {len(self.df)} games")
        self.report_lines.append(f"Date range: {self.df['GAME_DATE'].min()} to {self.df['GAME_DATE'].max()}")
        self.report_lines.append(f"Total features available: {len(self.df.columns)}")
        
        self.models = {}
        self.results = {}
        self.feature_importance = {}



    def prepare_data(self, target, test_size=20):
        """
        Prepare features and target for ML
        
        Args:
            target: What to predict ('PTS', 'REB', or 'AST')
            test_size: Number of most recent games to use for testing
        """

        # Define feature columns to use
        # Exclude non-feature columns and the target itself
        exclude_cols = [
            'GAME_ID', 'GAME_DATE', 'MATCHUP', 'SEASON_ID', 
            'Team_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME',
            'PTS', 'REB', 'AST', 'WL', 'MIN',  # Raw stats (targets)
            'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',  # Raw shooting stats
            'OREB', 'DREB', 'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS',
            'VIDEO_AVAILABLE', 'opponent', 'win', 'SEASON'
        ]
        
        # Select feature columns
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        
        # Remove rows with missing values in features or target
        data = self.df[feature_cols + [target]].copy()
        data = data.dropna()
        
        
        # Time-based split (CRITICAL: do not use random split!)
        train_data = data.iloc[:-test_size]
        test_data = data.iloc[-test_size:]
        
        X_train = train_data[feature_cols]
        y_train = train_data[target]
        X_test = test_data[feature_cols]
        y_test = test_data[target]
        

        # Store for later use
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_cols = feature_cols
        self.target = target
        
        return X_train, X_test, y_train, y_test
    

    def create_baseline(self):
        """
        Create a simple baseline: predict using last 5 games average
        """
        self.report_lines.append(f"\n{'='*70}")
        self.report_lines.append("BASELINE MODEL: Last 5 Games Average")
        self.report_lines.append('='*70)
        
        # Use the appropriate rolling average feature
        baseline_feature = f"{self.target.lower()}_last_5"
        
        if baseline_feature not in self.X_test.columns:
            self.report_lines.append(f"Warning: {baseline_feature} not found, using season average")
            baseline_feature = f"{self.target.lower()}_season_avg"
        
        baseline_predictions = self.X_test[baseline_feature].values
        
        mae = mean_absolute_error(self.y_test, baseline_predictions)
        rmse = np.sqrt(mean_squared_error(self.y_test, baseline_predictions))
        r2 = r2_score(self.y_test, baseline_predictions)
        
        self.report_lines.append(f"\nBaseline Performance:")
        self.report_lines.append(f"  MAE:  {mae:.2f} {self.target.lower()}")
        self.report_lines.append(f"  RMSE: {rmse:.2f} {self.target.lower()}")
        self.report_lines.append(f"  R²:   {r2:.3f}")
        
        self.baseline_mae = mae
        self.results['Baseline'] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': baseline_predictions
        }
        
        return mae

    def train_models(self):
        """
        Train multiple ML models
        """
        self.report_lines.append(f"\n{'='*70}")
        self.report_lines.append("TRAINING ML MODELS")
        self.report_lines.append('='*70)
        
        # Define models to try
        models_to_train = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        best_model = None
        best_improvement = -float('inf')
        
        for name, model in models_to_train.items():
            self.report_lines.append(f"\n{'-'*70}")
            self.report_lines.append(f"Training {name}...")
            
            # Train
            model.fit(self.X_train, self.y_train)
            
            # Predict
            train_pred = model.predict(self.X_train)
            test_pred = model.predict(self.X_test)
            
            # Evaluate
            train_mae = mean_absolute_error(self.y_train, train_pred)
            test_mae = mean_absolute_error(self.y_test, test_pred)
            test_rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
            test_r2 = r2_score(self.y_test, test_pred)
            
            self.report_lines.append(f"\n{name} Performance:")
            self.report_lines.append(f"  Train MAE:  {train_mae:.2f} {self.target.lower()}")
            self.report_lines.append(f"  Test MAE:   {test_mae:.2f} {self.target.lower()}")
            self.report_lines.append(f"  Test RMSE:  {test_rmse:.2f} {self.target.lower()}")
            self.report_lines.append(f"  Test R²:    {test_r2:.3f}")
            
            # Store results
            self.models[name] = model
            self.results[name] = {
                'mae': test_mae,
                'rmse': test_rmse,
                'r2': test_r2,
                'predictions': test_pred,
                'train_mae': train_mae
            }
            
            # Get feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = pd.DataFrame({
                    'feature': self.feature_cols,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Add top features to report
                top_features = self.feature_importance[name].head(10)
                self.report_lines.append(f"\n  Top 10 Most Important Features:")
                for idx, row in top_features.iterrows():
                    self.report_lines.append(f"    {row['feature']}: {row['importance']:.4f}")

            # Compare to baseline
            improvement = ((self.baseline_mae - test_mae) / self.baseline_mae) * 100
            if improvement > 0:
                self.report_lines.append(f"  ✓ {improvement:.1f}% better than baseline")
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_model = model
            else:
                self.report_lines.append(f"  {abs(improvement):.1f}% worse than baseline")
        
        self.best_model = best_model
        self.best_improvement = best_improvement if best_model is not None else None
        
        return best_model
    
    def generate_report(self):
        """
        Generate a formatted text report from all collected information
        Builds a fresh report from the collected data (doesn't use report_lines)
        """
        # Build report sections from scratch
        report_sections = []
        
        # Header
        report_sections.append("=" * 80)
        report_sections.append("NBA PLAYER PERFORMANCE PREDICTION - MODEL TRAINING REPORT")
        report_sections.append("=" * 80)
        
        # Dataset Information
        report_sections.append("\n" + "─" * 80)
        report_sections.append("DATASET INFORMATION")
        report_sections.append("─" * 80)
        report_sections.append(f"  Total Games Loaded:        {len(self.df)}")
        report_sections.append(f"  Date Range:                {self.df['GAME_DATE'].min().date()} to {self.df['GAME_DATE'].max().date()}")
        report_sections.append(f"  Total Features Available:  {len(self.df.columns)}")
        
        # Training Configuration
        report_sections.append("\n" + "─" * 80)
        report_sections.append("TRAINING CONFIGURATION")
        report_sections.append("─" * 80)
        report_sections.append(f"  Target Variable:           {self.target}")
        report_sections.append(f"  Training Samples:          {len(self.X_train)}")
        report_sections.append(f"  Test Samples:              {len(self.X_test)}")
        report_sections.append(f"  Features Used:             {len(self.feature_cols)}")
        
        # Baseline Results
        report_sections.append("\n" + "─" * 80)
        report_sections.append("BASELINE MODEL RESULTS")
        report_sections.append("─" * 80)
        baseline_result = self.results.get('Baseline', {})
        if baseline_result:
            report_sections.append(f"  Method:                    Last 5 Games Average")
            report_sections.append(f"  Mean Absolute Error (MAE): {baseline_result['mae']:.2f} {self.target.lower()}")
            report_sections.append(f"  Root Mean Squared Error:    {baseline_result['rmse']:.2f} {self.target.lower()}")
            report_sections.append(f"  R² Score:                  {baseline_result['r2']:.3f}")
        
        # Model Comparison Table
        report_sections.append("\n" + "─" * 80)
        report_sections.append("MODEL COMPARISON")
        report_sections.append("─" * 80)
        
        # Create formatted table with proper alignment
        table_header = f"{'Model':<22} │ {'MAE':>10} │ {'RMSE':>10} │ {'R²':>10} │ {'vs Baseline':>12}"
        report_sections.append(table_header)
        report_sections.append("─" * 80)
        
        for name, result in self.results.items():
            if isinstance(result, dict) and 'mae' in result:
                improvement = ((self.baseline_mae - result['mae']) / self.baseline_mae) * 100
                if name == 'Baseline':
                    improvement_str = "N/A"
                else:
                    improvement_str = f"{improvement:+.1f}%"
                
                table_row = (
                    f"{name:<22} │ "
                    f"{result['mae']:>10.2f} │ "
                    f"{result['rmse']:>10.2f} │ "
                    f"{result['r2']:>10.3f} │ "
                    f"{improvement_str:>12}"
                )
                report_sections.append(table_row)
        
        # Best Model Summary
        if hasattr(self, 'best_improvement') and self.best_improvement is not None:
            report_sections.append("\n" + "─" * 80)
            report_sections.append("BEST MODEL")
            report_sections.append("─" * 80)
            report_sections.append(f"  Best Improvement:          {self.best_improvement:.1f}% over baseline")
        
        # Feature Importance (if available)
        if self.feature_importance:
            report_sections.append("\n" + "─" * 80)
            report_sections.append("TOP FEATURES BY IMPORTANCE")
            report_sections.append("─" * 80)
            
            # Get the best tree-based model's features
            for model_name, feature_df in self.feature_importance.items():
                if len(feature_df) > 0:
                    report_sections.append(f"\n  {model_name}:")
                    top_features = feature_df.head(10)
                    for idx, row in top_features.iterrows():
                        report_sections.append(f"    • {row['feature']:<40} {row['importance']:>8.4f}")
                    break  # Only show first model's features
        
        report_sections.append("\n" + "=" * 80)
        report_sections.append("End of Report")
        report_sections.append("=" * 80)
        
        return "\n".join(report_sections)


def train_and_evaluate(features_csv, target='PTS', test_size=20):
    """
    Complete pipeline: train models and show results
    
    Args:
        features_csv: Path to the features CSV file
        target: What to predict ('PTS', 'REB', or 'AST')
        test_size: Number of recent games to use for testing
    """
    # Initialize
    predictor = NBAPredictor(features_csv)
    
    # Prepare data
    predictor.prepare_data(target=target, test_size=test_size)
    
    # Create baseline
    predictor.create_baseline()
    
    # Train models
    predictor.train_models()
    
    # Generate report
    predictor.generate_report()
    
    return predictor

