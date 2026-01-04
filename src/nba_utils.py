"""
NBA API utility functions for fetching upcoming games and player info
"""

from nba_api.stats.endpoints import teamgamelog, commonplayerinfo, leaguegamefinder, scoreboardv2
from nba_api.stats.static import players, teams
import pandas as pd
from datetime import datetime, timedelta
import time


def get_player_info(player_name):
    """Get player information including team"""
    try:
        player_list = players.find_players_by_full_name(player_name)
        if not player_list:
            return None
        
        player_id = player_list[0]['id']
        player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
        info_df = player_info.get_data_frames()[0]
        
        if len(info_df) == 0:
            return None
        
        team_id = info_df.iloc[0]['TEAM_ID'] if 'TEAM_ID' in info_df.columns else None
        team_abbrev = info_df.iloc[0]['TEAM_ABBREVIATION'] if 'TEAM_ABBREVIATION' in info_df.columns else None
        
        # If player is not on a team (retired/free agent), return None for team info
        if pd.isna(team_id) or team_id == 0:
            return {
                'id': player_id,
                'name': player_list[0]['full_name'],
                'team_id': None,
                'team_abbreviation': None
            }
        
        return {
            'id': player_id,
            'name': player_list[0]['full_name'],
            'team_id': int(team_id),
            'team_abbreviation': team_abbrev
        }
    except Exception as e:
        return None


def get_upcoming_games_for_team(team_id, days_ahead=30):
    """
    Get upcoming games for a team using multiple methods
    Tries ScoreboardV2 first (for today's games), then leaguegamefinder for future dates
    
    Args:
        team_id: Team ID
        days_ahead: Number of days to look ahead (default: 30)
    """
    try:
        today = datetime.now().date()
        end_date = today + timedelta(days=days_ahead)
        
        # Get team abbreviation
        team_abbrev = None
        for team in teams.get_teams():
            if team['id'] == team_id:
                team_abbrev = team['abbreviation']
                break
        
        if not team_abbrev:
            return pd.DataFrame()
        
        upcoming_games = []
        
        # Method 1: Try ScoreboardV2 for today and next few days
        try:
            for day_offset in range(min(7, days_ahead + 1)):  # Check next 7 days
                check_date = today + timedelta(days=day_offset)
                if check_date > end_date:
                    break
                
                try:
                    scoreboard = scoreboardv2.ScoreboardV2(
                        game_date=check_date.strftime('%m/%d/%Y'),
                        league_id='00',
                        day_offset='0'
                    )
                    
                    # Get game header data
                    game_header = scoreboard.get_data_frames()[0]
                    
                    if len(game_header) > 0:
                        for _, game in game_header.iterrows():
                            home_team_id = game.get('HOME_TEAM_ID')
                            visitor_team_id = game.get('VISITOR_TEAM_ID')
                            
                            # Check if our team is playing
                            if home_team_id == team_id or visitor_team_id == team_id:
                                is_home = home_team_id == team_id
                                opponent_id = visitor_team_id if is_home else home_team_id
                                
                                # Get opponent abbreviation
                                opponent_abbrev = None
                                for team in teams.get_teams():
                                    if team['id'] == opponent_id:
                                        opponent_abbrev = team['abbreviation']
                                        break
                                
                                if opponent_abbrev:
                                    upcoming_games.append({
                                        'date': check_date,
                                        'opponent_id': opponent_id,
                                        'opponent_abbrev': opponent_abbrev.upper(),
                                        'is_home': is_home,
                                        'game_id': game.get('GAME_ID', None)
                                    })
                    
                    time.sleep(0.3)  # Rate limiting
                except:
                    # Scoreboard might not have data for this date
                    continue
        except:
            pass
        
        # Method 2: If no games found, try leaguegamefinder with date range
        if len(upcoming_games) == 0:
            try:
                # Get current season
                current_year = datetime.now().year
                current_month = datetime.now().month
                
                if current_month >= 10:
                    season = f"{current_year}-{str(current_year + 1)[2:]}"
                else:
                    season = f"{current_year - 1}-{str(current_year)[2:]}"
                
                # Try to get games using leaguegamefinder
                # Note: This may not return future games, but worth trying
                game_finder = leaguegamefinder.LeagueGameFinder(
                    season_nullable=season,
                    league_id_nullable='00',
                    season_type_nullable='Regular Season'
                )
                
                games_df = game_finder.get_data_frames()[0]
                
                if len(games_df) > 0:
                    # Filter for our team
                    team_games = games_df[
                        (games_df['TEAM_ID'] == team_id) | 
                        (games_df['OPPONENT_TEAM_ID'] == team_id)
                    ].copy()
                    
                    if len(team_games) > 0:
                        # Convert GAME_DATE
                        team_games['GAME_DATE'] = pd.to_datetime(team_games['GAME_DATE'])
                        team_games = team_games[team_games['GAME_DATE'].dt.date > today]
                        team_games = team_games[team_games['GAME_DATE'].dt.date <= end_date]
                        team_games = team_games.sort_values('GAME_DATE')
                        
                        for _, game in team_games.iterrows():
                            is_home = game['TEAM_ID'] == team_id
                            opponent_id = game['OPPONENT_TEAM_ID'] if is_home else game['TEAM_ID']
                            
                            # Get opponent abbreviation
                            opponent_abbrev = None
                            for team in teams.get_teams():
                                if team['id'] == opponent_id:
                                    opponent_abbrev = team['abbreviation']
                                    break
                            
                            if opponent_abbrev:
                                # Check if we already have this game
                                game_date = game['GAME_DATE'].date()
                                if not any(g['date'] == game_date and g['opponent_abbrev'] == opponent_abbrev 
                                          for g in upcoming_games):
                                    upcoming_games.append({
                                        'date': game_date,
                                        'opponent_id': opponent_id,
                                        'opponent_abbrev': opponent_abbrev.upper(),
                                        'is_home': is_home,
                                        'game_id': game.get('GAME_ID', None)
                                    })
                
                time.sleep(0.6)  # Rate limiting
            except:
                pass
        
        # Remove duplicates and sort
        if upcoming_games:
            # Remove duplicates based on date and opponent
            seen = set()
            unique_games = []
            for game in upcoming_games:
                key = (game['date'], game['opponent_abbrev'])
                if key not in seen:
                    seen.add(key)
                    unique_games.append(game)
            
            # Sort by date
            unique_games.sort(key=lambda x: x['date'])
            
            return pd.DataFrame(unique_games)
        
        return pd.DataFrame()
    
    except Exception as e:
        # Silently return empty - upcoming games may not be available via API
        return pd.DataFrame()


def get_next_game_info(player_name, historical_data_df):
    """
    Get information about the next game for a player
    
    Args:
        player_name: Player name
        historical_data_df: DataFrame with historical game data (to calculate days_rest)
    """
    try:
        # Get player info
        player_info = get_player_info(player_name)
        if not player_info:
            return None
        
        if not player_info.get('team_id'):
            # Player might be retired, free agent, or not on a team
            return None
        
        # Get upcoming games
        upcoming = get_upcoming_games_for_team(player_info['team_id'], days_ahead=14)
        
        if len(upcoming) == 0:
            return None
        
        # Get the next game
        next_game = upcoming.iloc[0]
        
        # Calculate days rest from last game
        if len(historical_data_df) > 0:
            last_game_date = pd.to_datetime(historical_data_df['GAME_DATE'].iloc[-1]).date()
            next_game_date = next_game['date']
            days_rest = (next_game_date - last_game_date).days - 1
            days_rest = max(0, days_rest)  # Ensure non-negative
        else:
            days_rest = 1  # Default
        
        return {
            'date': next_game['date'],
            'opponent': next_game['opponent_abbrev'],
            'is_home': next_game['is_home'],
            'days_rest': days_rest,
            'opponent_id': next_game['opponent_id']
        }
    except Exception as e:
        return None


def get_all_upcoming_games(player_name, historical_data_df, num_games=5):
    """
    Get multiple upcoming games for a player
    
    Args:
        player_name: Player name
        historical_data_df: DataFrame with historical game data
        num_games: Number of upcoming games to return
    """
    # Get player info
    player_info = get_player_info(player_name)
    if not player_info or not player_info['team_id']:
        return []
    
    # Get upcoming games
    upcoming = get_upcoming_games_for_team(player_info['team_id'], days_ahead=30)
    
    if len(upcoming) == 0:
        return []
    
    # Limit to requested number
    upcoming = upcoming.head(num_games)
    
    # Calculate days rest for each game
    games_info = []
    last_game_date = None
    
    if len(historical_data_df) > 0:
        last_game_date = pd.to_datetime(historical_data_df['GAME_DATE'].iloc[-1]).date()
    
    for idx, game in upcoming.iterrows():
        if last_game_date:
            days_rest = (game['date'] - last_game_date).days - 1
            days_rest = max(0, days_rest)
            last_game_date = game['date']  # Update for next game
        else:
            days_rest = 1
        
        games_info.append({
            'date': game['date'],
            'opponent': game['opponent_abbrev'],
            'is_home': game['is_home'],
            'days_rest': days_rest,
            'opponent_id': game['opponent_id']
        })
    
    return games_info

