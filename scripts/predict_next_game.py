"""
Predict Upcoming NBA Games
This script uses your trained model to predict future games
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime

class GamePredictor:
    """
    Make predictions for upcoming games
    """
    
    def __init__(self, features_csv, model_path=None):
        """
        Initialize with historical data
        """
        self.df = pd.read_csv(features_csv)
        self.df['GAME_DATE'] = pd.to_datetime(self.df['GAME_DATE'])
        self.df = self.df.sort_values('GAME_DATE').reset_index(drop=True)
        
        # Load trained model if provided
        self.model = None
        if model_path:
            self.load_model(model_path)
    
    
    def load_model(self, model_path):
        """Load a pre-trained model"""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
    
    
    def create_prediction_features(self, opponent, is_home, days_rest):
        """
        Create features for an upcoming game based on recent history
        
        Args:
            opponent: Team abbreviation (e.g., 'LAL', 'BOS')
            is_home: True if home game, False if away
            days_rest: Days since last game (0, 1, 2, etc.)
        """
        # Get the most recent games for calculating features
        recent_games = self.df.tail(10)
        
        # Calculate features based on recent history
        features = {}
        
        # Rolling averages (based on most recent games)
        features['pts_last_3'] = recent_games.tail(3)['PTS'].mean()
        features['pts_last_5'] = recent_games.tail(5)['PTS'].mean()
        features['pts_last_10'] = recent_games['PTS'].mean()
        
        features['reb_last_3'] = recent_games.tail(3)['REB'].mean()
        features['reb_last_5'] = recent_games.tail(5)['REB'].mean()
        features['reb_last_10'] = recent_games['REB'].mean()
        
        features['ast_last_3'] = recent_games.tail(3)['AST'].mean()
        features['ast_last_5'] = recent_games.tail(5)['AST'].mean()
        features['ast_last_10'] = recent_games['AST'].mean()
        
        features['min_last_5'] = recent_games.tail(5)['MIN'].mean()
        features['min_last_10'] = recent_games['MIN'].mean()
        
        # Season averages (use all available games)
        features['pts_season_avg'] = self.df['PTS'].mean()
        features['reb_season_avg'] = self.df['REB'].mean()
        features['ast_season_avg'] = self.df['AST'].mean()
        features['min_season_avg'] = self.df['MIN'].mean()
        
        # Shooting percentages
        if 'FG_PCT' in recent_games.columns:
            features['fg_pct_last_5'] = recent_games.tail(5)['FG_PCT'].mean()
            features['fg_pct_last_10'] = recent_games['FG_PCT'].mean()
            features['fg_pct_season_avg'] = self.df['FG_PCT'].mean()
        
        if 'FG3_PCT' in recent_games.columns:
            features['fg3_pct_last_5'] = recent_games.tail(5)['FG3_PCT'].mean()
            features['fg3_pct_last_10'] = recent_games['FG3_PCT'].mean()
            features['fg3_pct_season_avg'] = self.df['FG3_PCT'].mean()
        
        if 'FT_PCT' in recent_games.columns:
            features['ft_pct_last_5'] = recent_games.tail(5)['FT_PCT'].mean()
            features['ft_pct_last_10'] = recent_games['FT_PCT'].mean()
            features['ft_pct_season_avg'] = self.df['FT_PCT'].mean()
        
        # Rest & Schedule features
        features['days_rest'] = days_rest
        features['is_back_to_back'] = 1 if days_rest == 0 else 0
        features['is_2nd_night_b2b'] = 0  # Would need more context
        
        # Calculate games in last 7 days
        # Estimate based on days_rest
        if days_rest == 0:
            features['games_in_last_7_days'] = 4  # Estimate for back-to-back
        elif days_rest == 1:
            features['games_in_last_7_days'] = 3
        else:
            features['games_in_last_7_days'] = 2
        
        features['game_num_season'] = len(self.df) + 1
        
        # Home/Away features
        features['is_home'] = 1 if is_home else 0
        
        # Home/away splits (calculate from historical data)
        home_games = self.df[self.df['is_home'] == 1]
        away_games = self.df[self.df['is_home'] == 0]
        
        if len(home_games) > 0:
            features['pts_home_avg'] = home_games['PTS'].mean()
            features['reb_home_avg'] = home_games['REB'].mean()
            features['ast_home_avg'] = home_games['AST'].mean()
        else:
            features['pts_home_avg'] = self.df['PTS'].mean()
            features['reb_home_avg'] = self.df['REB'].mean()
            features['ast_home_avg'] = self.df['AST'].mean()
        
        if len(away_games) > 0:
            features['pts_away_avg'] = away_games['PTS'].mean()
            features['reb_away_avg'] = away_games['REB'].mean()
            features['ast_away_avg'] = away_games['AST'].mean()
        else:
            features['pts_away_avg'] = self.df['PTS'].mean()
            features['reb_away_avg'] = self.df['REB'].mean()
            features['ast_away_avg'] = self.df['AST'].mean()
        
        # Trend features (calculate slope of last 5 games)
        last_5_pts = recent_games.tail(5)['PTS'].values
        if len(last_5_pts) >= 2:
            x = np.arange(len(last_5_pts))
            features['pts_trend'] = np.polyfit(x, last_5_pts, 1)[0]
        else:
            features['pts_trend'] = 0
        
        # Consistency features
        features['pts_std_10'] = recent_games['PTS'].std()
        features['reb_std_10'] = recent_games['REB'].std()
        features['ast_std_10'] = recent_games['AST'].std()
        
        features['pts_cv_10'] = features['pts_std_10'] / features['pts_last_10'] if features['pts_last_10'] > 0 else 0
        
        # Opponent history (head-to-head)
        opp_games = self.df[self.df['opponent'] == opponent]
        if len(opp_games) >= 2:
            features['pts_vs_opp_avg'] = opp_games['PTS'].mean()
            features['reb_vs_opp_avg'] = opp_games['REB'].mean()
            features['ast_vs_opp_avg'] = opp_games['AST'].mean()
            features['pts_last_vs_opp'] = opp_games.iloc[-1]['PTS']
        else:
            # No history vs this opponent
            features['pts_vs_opp_avg'] = features['pts_season_avg']
            features['reb_vs_opp_avg'] = features['reb_season_avg']
            features['ast_vs_opp_avg'] = features['ast_season_avg']
            features['pts_last_vs_opp'] = features['pts_last_5']
        
        # Additional features that might be in your model
        if 'STL' in recent_games.columns:
            features['stl_last_5'] = recent_games.tail(5)['STL'].mean()
            features['stl_last_10'] = recent_games['STL'].mean()
            features['stl_season_avg'] = self.df['STL'].mean()
        
        if 'BLK' in recent_games.columns:
            features['blk_last_5'] = recent_games.tail(5)['BLK'].mean()
            features['blk_last_10'] = recent_games['BLK'].mean()
            features['blk_season_avg'] = self.df['BLK'].mean()
        
        if 'TOV' in recent_games.columns:
            features['tov_last_5'] = recent_games.tail(5)['TOV'].mean()
            features['tov_last_10'] = recent_games['TOV'].mean()
            features['tov_season_avg'] = self.df['TOV'].mean()
        
        # Additional trend features
        if 'REB' in recent_games.columns:
            last_5_reb = recent_games.tail(5)['REB'].values
            if len(last_5_reb) >= 2:
                features['reb_trend'] = np.polyfit(np.arange(len(last_5_reb)), last_5_reb, 1)[0]
            else:
                features['reb_trend'] = 0
        
        if 'AST' in recent_games.columns:
            last_5_ast = recent_games.tail(5)['AST'].values
            if len(last_5_ast) >= 2:
                features['ast_trend'] = np.polyfit(np.arange(len(last_5_ast)), last_5_ast, 1)[0]
            else:
                features['ast_trend'] = 0
        
        if 'MIN' in recent_games.columns:
            last_5_min = recent_games.tail(5)['MIN'].values
            if len(last_5_min) >= 2:
                features['min_trend'] = np.polyfit(np.arange(len(last_5_min)), last_5_min, 1)[0]
            else:
                features['min_trend'] = 0
        
        # Consistency for REB and AST
        if 'reb_last_10' in features and features['reb_last_10'] > 0:
            features['reb_cv_10'] = features['reb_std_10'] / features['reb_last_10']
        else:
            features['reb_cv_10'] = 0
            
        if 'ast_last_10' in features and features['ast_last_10'] > 0:
            features['ast_cv_10'] = features['ast_std_10'] / features['ast_last_10']
        else:
            features['ast_cv_10'] = 0
        
        return features
    
    
    def predict_game(self, opponent, is_home=True, days_rest=1):
        """
        Predict points, rebounds, and assists for upcoming game
        
        Args:
            opponent: Team abbreviation (e.g., 'LAL', 'GSW', 'BOS')
            is_home: True if home game, False if away
            days_rest: Days since last game (0=back-to-back, 1, 2, etc.)
        """
        # Create features
        features = self.create_prediction_features(opponent, is_home, days_rest)
        
        # Start with last 5 average
        pts_pred = features['pts_last_5']
        reb_pred = features['reb_last_5']
        ast_pred = features['ast_last_5']
        
        # Adjust for home/away
        if is_home:
            home_boost = features['pts_home_avg'] - features['pts_season_avg']
            pts_pred += home_boost * 0.5  # Apply 50% of historical home advantage
        else:
            away_penalty = features['pts_season_avg'] - features['pts_away_avg']
            pts_pred -= away_penalty * 0.5
        
        # Adjust for rest
        if days_rest == 0:
            pts_pred *= 0.93  # Back-to-back penalty (~7% reduction)
            reb_pred *= 0.95
            ast_pred *= 0.95
        
        # Adjust for opponent history
        if features['pts_vs_opp_avg'] != features['pts_season_avg']:
            opp_factor = (features['pts_vs_opp_avg'] - features['pts_season_avg']) * 0.3
            pts_pred += opp_factor
        
        return {
            'points': pts_pred,
            'rebounds': reb_pred,
            'assists': ast_pred,
            'features': features
        }


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # Look for features CSV
        import os
        csv_files = [f for f in os.listdir('.') if f.endswith('_features.csv')]
        
        if csv_files:
            csv_file = csv_files[0]
        else:
            sys.exit(1)
    
    # Initialize predictor
    predictor = GamePredictor(csv_file)
    
    # Get game details from user
    opponent = input("\nOpponent team abbreviation (e.g., LAL, GSW, BOS): ").strip().upper()
    
    location_input = input("Home or Away? (H/A) [default: H]: ").strip().upper()
    is_home = location_input != 'A'
    
    rest_input = input("Days since last game (0=back-to-back, 1, 2, etc.) [default: 1]: ").strip()
    days_rest = int(rest_input) if rest_input else 1
    
    # Make prediction
    prediction = predictor.predict_game(opponent, is_home, days_rest)
