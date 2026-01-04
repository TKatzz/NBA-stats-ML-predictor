"""
Feature Engineering for Single Player Performance Prediction
This script takes raw player game logs and creates ML-ready features
"""

import pandas as pd
import numpy as np
from datetime import datetime
from nba_api.stats.endpoints import teamdashboardbygeneralsplits, leaguedashteamstats
from nba_api.stats.static import teams
import time

class PlayerFeatureEngineer:
    """
    Creates features for predicting a single player's performance
    """
    
    def __init__(self, player_data_df):
        """
        Initialize with player's game log CSV
        """
        self.df = player_data_df
        self.df['GAME_DATE'] = pd.to_datetime(self.df['GAME_DATE'])
        self.df = self.df.sort_values('GAME_DATE').reset_index(drop=True)
        
        print(f"Loaded {len(self.df)} games")
        print(f"Date range: {self.df['GAME_DATE'].min()} to {self.df['GAME_DATE'].max()}")
    
    
    def create_basic_features(self):
        
        # Home vs Away
        self.df['is_home'] = self.df['MATCHUP'].str.contains('vs.').astype(int)
        
        # Extract opponent team abbreviation
        self.df['opponent'] = self.df['MATCHUP'].str.extract(r'(vs\.|@)\s+([A-Z]{3})')[1]
        
        # Win/Loss to binary
        if 'WL' in self.df.columns:
            self.df['win'] = (self.df['WL'] == 'W').astype(int)
        
        return self
    
    
    def create_rest_features(self):
        """
        Create rest and scheduling features
        """
        # Days between games
        self.df['days_rest'] = self.df['GAME_DATE'].diff().dt.days - 1
        self.df['days_rest'] = self.df['days_rest'].fillna(3)  # First game, assume normal rest
        
        # Back-to-back games
        self.df['is_back_to_back'] = (self.df['days_rest'] == 0).astype(int)
        
        # Second night of back-to-back (even more tiring)
        self.df['is_2nd_night_b2b'] = (
            (self.df['is_back_to_back'] == 1) & 
            (self.df['is_back_to_back'].shift(1) == 1)
        ).astype(int)
        
        # Games in last 7 days (fatigue measure)
        self.df['games_in_last_7_days'] = self.df['GAME_DATE'].rolling(window=7).count() - 1  # -1 to exclude current game
        
        # Game number in season
        if 'SEASON' in self.df.columns:
            self.df['game_num_season'] = self.df.groupby('SEASON').cumcount() + 1
        else:
            self.df['game_num_season'] = range(1, len(self.df) + 1)
        
        return self
    
    
    def create_rolling_averages(self, windows=[3, 5, 10]):
        """
        Create rolling average features for key stats
        CRITICAL: Use .shift(1) to avoid data leakage!
        """
        print(f"\n3. Creating rolling averages (windows: {windows})...")
        
        key_stats = ['PTS', 'REB', 'AST', 'MIN', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'STL', 'BLK', 'TOV']
        available_stats = [stat for stat in key_stats if stat in self.df.columns]
        
        for window in windows:
            for stat in available_stats:
                col_name = f"{stat.lower()}_last_{window}"
                
                # .shift(1) ensures we don't use the current game in the average
                self.df[col_name] = self.df[stat].rolling(window=window, min_periods=1).mean().shift(1)
        
        # Season averages (expanding mean)
        for stat in available_stats:
            col_name = f"{stat.lower()}_season_avg"
            self.df[col_name] = self.df[stat].expanding(min_periods=1).mean().shift(1)
        
        
        return self
    
    
    def create_trend_features(self, window=5):
        """
        Create trend features (is performance improving or declining?)
        """

        key_stats = ['PTS', 'REB', 'AST', 'MIN']
        available_stats = [stat for stat in key_stats if stat in self.df.columns]
        
        for stat in available_stats:
            col_name = f"{stat.lower()}_trend"
            
            # Calculate slope of last N games
            # Positive slope = improving, negative = declining
            def calculate_trend(series):
                if len(series) < 2:
                    return 0
                x = np.arange(len(series))
                y = series.values
                # Simple linear regression slope
                slope = np.polyfit(x, y, 1)[0]
                return slope
            
            self.df[col_name] = self.df[stat].rolling(window=window).apply(
                calculate_trend, raw=False
            ).shift(1)
        
        return self
    
    
    def create_consistency_features(self, window=10):
        """
        Create consistency/variance features
        """
        
        key_stats = ['PTS', 'REB', 'AST']
        available_stats = [stat for stat in key_stats if stat in self.df.columns]
        
        for stat in available_stats:
            # Standard deviation
            col_name_std = f"{stat.lower()}_std_{window}"
            self.df[col_name_std] = self.df[stat].rolling(window=window, min_periods=2).std().shift(1)
            
            # Coefficient of variation (std / mean)
            col_name_cv = f"{stat.lower()}_cv_{window}"
            rolling_mean = self.df[stat].rolling(window=window, min_periods=2).mean().shift(1)
            rolling_std = self.df[col_name_std]
            self.df[col_name_cv] = rolling_std / rolling_mean.replace(0, np.nan)
        
        
        return self
    
    
    def create_home_away_splits(self):
        """
        Create home/away performance averages
        """
        print("\n6. Creating home/away split features...")
        
        key_stats = ['PTS', 'REB', 'AST']
        available_stats = [stat for stat in key_stats if stat in self.df.columns]
        
        for stat in available_stats:
            # Calculate expanding averages for home and away separately
            home_avg = self.df[self.df['is_home'] == 1].groupby(
                self.df[self.df['is_home'] == 1].index
            )[stat].expanding().mean().reset_index(level=0, drop=True)
            
            away_avg = self.df[self.df['is_home'] == 0].groupby(
                self.df[self.df['is_home'] == 0].index
            )[stat].expanding().mean().reset_index(level=0, drop=True)
            
            # Map back to full dataframe
            self.df[f"{stat.lower()}_home_avg"] = np.nan
            self.df[f"{stat.lower()}_away_avg"] = np.nan
            
            self.df.loc[self.df['is_home'] == 1, f"{stat.lower()}_home_avg"] = home_avg.shift(1)
            self.df.loc[self.df['is_home'] == 0, f"{stat.lower()}_away_avg"] = away_avg.shift(1)
            
            # Fill forward for opposite location
            self.df[f"{stat.lower()}_home_avg"] = self.df[f"{stat.lower()}_home_avg"].ffill()
            self.df[f"{stat.lower()}_away_avg"] = self.df[f"{stat.lower()}_away_avg"].ffill()
        
        print(f"   ✓ Added home/away averages for: {', '.join(available_stats)}")
        
        return self
    
    
    def create_opponent_history(self, min_games=2):
        """
        Create head-to-head features (performance vs specific opponent)
        """
        print("\n7. Creating opponent history features...")
        
        key_stats = ['PTS', 'REB', 'AST']
        available_stats = [stat for stat in key_stats if stat in self.df.columns]
        
        for stat in available_stats:
            col_name = f"{stat.lower()}_vs_opp_avg"
            self.df[col_name] = np.nan
            
            # For each game, calculate average vs this opponent (excluding current game)
            for idx in self.df.index:
                opponent = self.df.loc[idx, 'opponent']
                
                # Get all previous games vs this opponent
                prev_games = self.df[
                    (self.df.index < idx) & 
                    (self.df['opponent'] == opponent)
                ]
                
                if len(prev_games) >= min_games:
                    self.df.loc[idx, col_name] = prev_games[stat].mean()
        
        # Last game vs this opponent
        self.df['pts_last_vs_opp'] = np.nan
        for idx in self.df.index:
            opponent = self.df.loc[idx, 'opponent']
            prev_games = self.df[
                (self.df.index < idx) & 
                (self.df['opponent'] == opponent)
            ]
            if len(prev_games) > 0:
                self.df.loc[idx, 'pts_last_vs_opp'] = prev_games.iloc[-1]['PTS']
        
        print(f"   ✓ Added opponent history for: {', '.join(available_stats)}")
        
        return self
    
    
    def add_opponent_team_stats(self, season='2023-24'):
        """
        Add opponent team defensive statistics
        This requires API calls, so it's optional but valuable
        """
        print(f"\n8. Fetching opponent team stats for {season}...")
        print("   (This may take a while...)")
        
        # Get all team stats
        try:
            team_stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                season_type_all_star='Regular Season',
                measure_type_detailed_defense='Defense'
            )
            team_df = team_stats.get_data_frames()[0]
            
            # Create mapping of team abbreviation to defensive stats
            team_map = {}
            teams_list = teams.get_teams()
            
            for _, row in team_df.iterrows():
                # Find team abbreviation
                team_name = row['TEAM_NAME']
                team_info = [t for t in teams_list if t['full_name'] == team_name]
                
                if team_info:
                    abbrev = team_info[0]['abbreviation']
                    team_map[abbrev] = {
                        'def_rating': row.get('DEF_RATING', None),
                        'opp_pts_pg': row.get('OPP_PTS', None),
                        'pace': row.get('PACE', None)
                    }
            
            # Map to dataframe
            self.df['opp_def_rating'] = self.df['opponent'].map(
                lambda x: team_map.get(x, {}).get('def_rating', None)
            )
            self.df['opp_pts_allowed'] = self.df['opponent'].map(
                lambda x: team_map.get(x, {}).get('opp_pts_pg', None)
            )
            self.df['opp_pace'] = self.df['opponent'].map(
                lambda x: team_map.get(x, {}).get('pace', None)
            )
            
            print("   ✓ Added: opp_def_rating, opp_pts_allowed, opp_pace")
            
        except Exception as e:
            print(f"   ✗ Error fetching team stats: {e}")
            print("   Continuing without opponent stats...")
        
        return self
    
    
    def get_feature_dataframe(self, drop_early_games=10):
        """
        Get the final feature dataframe ready for ML
        
        Args:
            drop_early_games: Drop first N games (not enough history for features)
        """
        print(f"\n9. Preparing final dataset...")
        print(f"   Dropping first {drop_early_games} games (insufficient history)")
        
        # Drop early games that don't have enough history
        df_final = self.df.iloc[drop_early_games:].copy()
        
        print(f"   Final dataset: {len(df_final)} games")
        print(f"   Total features: {len(df_final.columns)}")
        
        return df_final
    
    
    def save_features(self, filename='player_features_ml_ready.csv'):
        """
        Save engineered features to CSV
        """
        df_final = self.get_feature_dataframe()
        df_final.to_csv(filename, index=False)
        print(f"\n✓ Features saved to: {filename}")
        
        return df_final


def create_all_features(player_df_table, save_output=True):
    """
    One-function pipeline to create all features
    
    Usage:
        df = create_all_features('Stephen_Curry_data.csv')
    """

    # Initialize
    engineer = PlayerFeatureEngineer(player_df_table)
    
    # Create all features
    engineer.create_basic_features()
    engineer.create_rest_features()
    engineer.create_rolling_averages(windows=[3, 5, 10])
    engineer.create_trend_features(window=5)
    engineer.create_consistency_features(window=10)
    engineer.create_home_away_splits()
    engineer.create_opponent_history(min_games=2)
    
    # Optional: Add opponent stats (requires API calls)
    # Uncomment if you want opponent team stats
    # engineer.add_opponent_team_stats(season='2023-24')
    
    # Get final dataframe
    df_final = engineer.get_feature_dataframe(drop_early_games=10)
    

    return df_final



# Create all features
#df = create_all_features(csv_file, save_output=True)

