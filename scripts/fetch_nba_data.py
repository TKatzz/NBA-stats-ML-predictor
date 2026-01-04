"""
NBA data fetching modules 
"""


from nba_api.stats.endpoints import playergamelog, leaguegamefinder, commonplayerinfo, teamgamelog
from nba_api.stats.static import players, teams
import pandas as pd
import time



def get_active_players():
    """Get dataframe of all NBA players (active and inactive)"""
    active_players = players.get_active_players()
    df_players = pd.DataFrame(active_players)


    return df_players


def quick_fetch_player_data(player_name, seasons=['2023-24', '2024-25', '2025-26']):
    # Find player
    player_list = players.find_players_by_full_name(player_name)
    if not player_list:
        print(f"Player '{player_name}' not found!")
        return None
    
    player_id = player_list[0]['id']

    # Fetch data for each season
    all_games = []
    
    for season in seasons:
        
        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star='Regular Season'
        )
        
        df = gamelog.get_data_frames()[0]
        df['SEASON'] = season
        all_games.append(df)
        
        time.sleep(1)
    
    # Combine all seasons
    full_data = pd.concat(all_games, ignore_index=True)
    
    # Convert date to datetime
    full_data['GAME_DATE'] = pd.to_datetime(full_data['GAME_DATE'])
    
    # Sort by date
    full_data = full_data.sort_values('GAME_DATE').reset_index(drop=True)
    
    #print(f"\nTotal games fetched: {len(full_data)}")
    return full_data

