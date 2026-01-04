"""
NBA Player Performance Prediction - Streamlit Web Application
Main application file
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add parent directory to path to import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from scripts.fetch_nba_data import quick_fetch_player_data, get_active_players
from scripts.feature_engineering import create_all_features
from scripts.predict_next_game import GamePredictor
from scripts.model_train import train_and_evaluate
from src.nba_utils import get_next_game_info, get_all_upcoming_games, get_player_info
from nba_api.stats.static import teams

# Page configuration
st.set_page_config(
    page_title="NBA Player Performance Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background-color: #1e293b;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #3b82f6;
        margin: 1rem 0;
        color: #e2e8f0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 2rem;
        border-radius: 1rem;
        border: 1px solid #334155;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
        margin: 1.5rem 0;
        color: #e2e8f0;
    }
    .prediction-metric {
        background-color: #0f172a;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #334155;
        text-align: center;
        margin: 0.5rem;
    }
    .prediction-metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #60a5fa;
        margin: 0.5rem 0;
    }
    .prediction-metric-label {
        font-size: 1rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'player_data' not in st.session_state:
    st.session_state.player_data = None
if 'features_data' not in st.session_state:
    st.session_state.features_data = None
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'player_name' not in st.session_state:
    st.session_state.player_name = None
if 'show_manual_input' not in st.session_state:
    st.session_state.show_manual_input = False
if 'manual_opponent' not in st.session_state:
    st.session_state.manual_opponent = ""
if 'manual_is_home' not in st.session_state:
    st.session_state.manual_is_home = True
if 'manual_days_rest' not in st.session_state:
    st.session_state.manual_days_rest = 1
if 'trained_predictor' not in st.session_state:
    st.session_state.trained_predictor = None
if 'training_report' not in st.session_state:
    st.session_state.training_report = None


def main():
    # Header
    st.markdown('<h1 class="main-header">🏀 NBA Player Performance Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Player selection
        st.subheader("1. Select Player")
        
        # Get list of active players for dropdown
        @st.cache_data(ttl=3600)  # Cache for 1 hour
        def get_players_list():
            try:
                players_df = get_active_players()
                if players_df is not None and len(players_df) > 0:
                    # Create display names: "Full Name (Team)"
                    player_names = players_df['full_name'].tolist()
                    return sorted(player_names)
                return ["Stephen Curry", "LeBron James", "Kevin Durant", "Luka Doncic", "Jayson Tatum"]
            except:
                return ["Stephen Curry", "LeBron James", "Kevin Durant", "Luka Doncic", "Jayson Tatum"]
        
        player_list = get_players_list()
        default_index = 0
        if st.session_state.player_name and st.session_state.player_name in player_list:
            default_index = player_list.index(st.session_state.player_name)
        
        player_input = st.selectbox(
            "Player Name",
            options=player_list,
            index=default_index,
            help="Select an NBA player from the list"
        )
        
        # Season selection
        st.subheader("2. Select Seasons")
        available_seasons = [
            '2015-16', '2016-17', '2017-18', '2018-19', '2019-20',
            '2020-21', '2021-22', '2022-23', 
            '2023-24', '2024-25', '2025-26'
        ]
        selected_seasons = st.multiselect(
            "Seasons to include",
            options=available_seasons,
            default=['2023-24', '2024-25'],
            help="Select one or more seasons"
        )
        
        # Action buttons
        st.subheader("3. Actions")
        
        fetch_button = st.button("📥 Fetch Player Data", use_container_width=True)
        engineer_button = st.button("🔧 Create Features", use_container_width=True, 
                                   disabled=st.session_state.player_data is None)
        predict_button = st.button("🔮 Predict Next Game", use_container_width=True,
                                  disabled=st.session_state.features_data is None)
        
        # Clear data button
        if st.button("🗑️ Clear All Data", use_container_width=True):
            # Clean up temp file if it exists
            if 'temp_file' in st.session_state and os.path.exists(st.session_state.temp_file):
                try:
                    os.remove(st.session_state.temp_file)
                except:
                    pass
            
            st.session_state.player_data = None
            st.session_state.features_data = None
            st.session_state.predictor = None
            st.session_state.player_name = None
            st.session_state.next_game_info = None
            st.session_state.prediction = None
            st.session_state.show_manual_input = False
            st.session_state.manual_opponent = ""
            st.session_state.manual_is_home = True
            st.session_state.manual_days_rest = 1
            st.session_state.trained_predictor = None
            st.session_state.training_report = None
            if 'temp_file' in st.session_state:
                del st.session_state.temp_file
            st.rerun()
    
    # Main content area
    if fetch_button:
        if not player_input:
            st.error("Please enter a player name")
        elif not selected_seasons:
            st.error("Please select at least one season")
        else:
            with st.spinner(f"Fetching data for {player_input}..."):
                try:
                    df = quick_fetch_player_data(player_input, seasons=selected_seasons)
                    if df is not None and len(df) > 0:
                        st.session_state.player_data = df
                        st.session_state.player_name = player_input
                        st.success(f"✓ Successfully fetched {len(df)} games for {player_input}")
                        st.rerun()
                    else:
                        st.error(f"No data found for {player_input}")
                except Exception as e:
                    st.error(f"Error fetching data: {str(e)}")
    
    # Display player data
    if st.session_state.player_data is not None:
        st.header("📊 Player Game Data")
        
        df = st.session_state.player_data
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Games", len(df))
        with col2:
            st.metric("Avg Points", f"{df['PTS'].mean():.1f}")
        with col3:
            st.metric("Avg Rebounds", f"{df['REB'].mean():.1f}")
        with col4:
            st.metric("Avg Assists", f"{df['AST'].mean():.1f}")
        
        # Data table
        st.subheader("Game Log Table")
        
        # Column selection
        display_cols = st.multiselect(
            "Select columns to display",
            options=df.columns.tolist(),
            default=['GAME_DATE', 'MATCHUP', 'WL', 'MIN', 'PTS', 'REB', 'AST', 'FG_PCT', 'FG3_PCT', 'FT_PCT'],
            key="column_selector"
        )
        
        if display_cols:
            display_df = df[display_cols].copy()
            
            # Format date column
            if 'GAME_DATE' in display_df.columns:
                display_df['GAME_DATE'] = pd.to_datetime(display_df['GAME_DATE']).dt.strftime('%Y-%m-%d')
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
            # Export to CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv,
                file_name=f"{st.session_state.player_name.replace(' ', '_')}_data.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # Feature engineering
    if engineer_button and st.session_state.player_data is not None:
        with st.spinner("Creating features... This may take a moment..."):
            try:
                df_features = create_all_features(st.session_state.player_data, save_output=False)
                st.session_state.features_data = df_features
                # Reset predictor when features are regenerated
                if 'temp_file' in st.session_state and os.path.exists(st.session_state.temp_file):
                    try:
                        os.remove(st.session_state.temp_file)
                    except:
                        pass
                st.session_state.predictor = None
                if 'temp_file' in st.session_state:
                    del st.session_state.temp_file
                st.success(f"✓ Successfully created features for {len(df_features)} games")
                st.rerun()
            except Exception as e:
                st.error(f"Error creating features: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Display features
    if st.session_state.features_data is not None:
        st.header("🔧 Engineered Features")
        
        df_features = st.session_state.features_data
        
        st.metric("Total Features", len(df_features.columns))
        st.metric("Games with Features", len(df_features))
        
        # Show feature columns
        with st.expander("View Feature Columns"):
            st.write(df_features.columns.tolist())
        
        # Download features
        csv_features = df_features.to_csv(index=False)
        st.download_button(
            label="📥 Download Features as CSV",
            data=csv_features,
            file_name=f"{st.session_state.player_name.replace(' ', '_')}_features.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Prediction section
    if predict_button and st.session_state.features_data is not None:
        # Train models and generate report if not already done
        if st.session_state.trained_predictor is None:
            with st.spinner("Training models and generating report... This may take a moment..."):
                try:
                    import tempfile
                    temp_dir = tempfile.gettempdir()
                    temp_features_file = os.path.join(
                        temp_dir, 
                        f"temp_{st.session_state.player_name.replace(' ', '_')}_features.csv"
                    )
                    st.session_state.features_data.to_csv(temp_features_file, index=False)
                    
                    # Train models and generate report
                    trained_predictor = train_and_evaluate(temp_features_file, target='PTS', test_size=20)
                    report_text = trained_predictor.generate_report()
                    
                    st.session_state.trained_predictor = trained_predictor
                    st.session_state.training_report = report_text
                except Exception as e:
                    st.warning(f"Could not train models: {str(e)}")
        
        # Initialize predictor if needed
        if st.session_state.predictor is None:
            import tempfile
            temp_dir = tempfile.gettempdir()
            temp_features_file = os.path.join(
                temp_dir, 
                f"temp_{st.session_state.player_name.replace(' ', '_')}_features.csv"
            )
            st.session_state.features_data.to_csv(temp_features_file, index=False)
            st.session_state.predictor = GamePredictor(temp_features_file)
            st.session_state.temp_file = temp_features_file
        
        # Try to get next game from API
        next_game = None
        api_error = None
        with st.spinner("Getting next game information from NBA API..."):
            try:
                next_game = get_next_game_info(
                    st.session_state.player_name,
                    st.session_state.player_data
                )
                if next_game is None:
                    api_error = "No upcoming games found in the next 14 days. The NBA API may not return future scheduled games."
            except Exception as e:
                api_error = f"API Error: {str(e)}"
                import traceback
                st.session_state.api_error_details = traceback.format_exc()
        
        # If API doesn't return game, show manual input
        if not next_game:
            st.session_state.show_manual_input = True
            if api_error:
                st.warning(f"{api_error}")
            st.info("📝 Please enter game details manually:")
        else:
            # Make prediction with API data
            st.session_state.show_manual_input = False
            try:
                prediction = st.session_state.predictor.predict_game(
                    opponent=next_game['opponent'],
                    is_home=next_game['is_home'],
                    days_rest=next_game['days_rest']
                )
                
                st.session_state.next_game_info = next_game
                st.session_state.prediction = prediction
                st.rerun()
            except Exception as pred_error:
                st.error(f"Error making prediction: {str(pred_error)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Show manual input form (persistent)
    if st.session_state.show_manual_input and st.session_state.features_data is not None:
        # Initialize predictor if needed
        if st.session_state.predictor is None:
            import tempfile
            temp_dir = tempfile.gettempdir()
            temp_features_file = os.path.join(
                temp_dir, 
                f"temp_{st.session_state.player_name.replace(' ', '_')}_features.csv"
            )
            st.session_state.features_data.to_csv(temp_features_file, index=False)
            st.session_state.predictor = GamePredictor(temp_features_file)
            st.session_state.temp_file = temp_features_file
        
        st.markdown("---")
        st.subheader("📝 Manual Game Input")
        
        # Get list of NBA teams for dropdown
        @st.cache_data(ttl=3600)  # Cache for 1 hour
        def get_teams_list():
            try:
                nba_teams = teams.get_teams()
                # Create list of team abbreviations sorted alphabetically
                team_abbrevs = sorted([team['abbreviation'] for team in nba_teams])
                return team_abbrevs
            except:
                return ['ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 
                        'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NO', 'NYK', 
                        'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SA', 'TOR', 'UTA', 'WAS']
        
        team_list = get_teams_list()
        default_team_index = 0
        if st.session_state.manual_opponent and st.session_state.manual_opponent in team_list:
            default_team_index = team_list.index(st.session_state.manual_opponent)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            opponent_input = st.selectbox(
                "Opponent Team", 
                options=team_list,
                index=default_team_index,
                key="manual_opponent_input",
                help="Select the opponent team"
            )
        with col2:
            location_options = ["Home", "Away"]
            location_index = 0 if st.session_state.manual_is_home else 1
            is_home_input = st.selectbox(
                "Location", 
                location_options,
                index=location_index,
                key="manual_location_input"
            ) == "Home"
        with col3:
            days_rest_input = st.number_input(
                "Days Rest", 
                min_value=0, 
                max_value=7, 
                value=st.session_state.manual_days_rest,
                key="manual_rest_input",
                help="Days since last game (0 = back-to-back)"
            )
        
        # Update session state with current values
        st.session_state.manual_opponent = opponent_input
        st.session_state.manual_is_home = is_home_input
        st.session_state.manual_days_rest = days_rest_input
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🔮 Make Prediction", use_container_width=True, type="primary"):
                if opponent_input:
                    try:
                        prediction = st.session_state.predictor.predict_game(
                            opponent=opponent_input.upper(),
                            is_home=is_home_input,
                            days_rest=days_rest_input
                        )
                        
                        st.session_state.next_game_info = {
                            'date': 'TBD',
                            'opponent': opponent_input.upper(),
                            'is_home': is_home_input,
                            'days_rest': days_rest_input
                        }
                        st.session_state.prediction = prediction
                        st.session_state.show_manual_input = False
                        st.rerun()
                    except Exception as pred_error:
                        st.error(f"Error making prediction: {str(pred_error)}")
                        import traceback
                        st.code(traceback.format_exc())
        
        with col2:
            if st.button("Cancel Manual Input", use_container_width=True):
                st.session_state.show_manual_input = False
                st.session_state.manual_opponent = ""
                st.rerun()
    
    # Display prediction
    if 'prediction' in st.session_state and st.session_state.prediction is not None:
        st.header("🔮 Next Game Prediction")
        
        next_game = st.session_state.next_game_info
        prediction = st.session_state.prediction
        
        # Game info card
        location = "vs" if next_game['is_home'] else "@"
        st.markdown(f"""
        <div class="prediction-box">
            <h3 style="color: #e2e8f0; margin-top: 0;">Upcoming Game: {location} {next_game['opponent']}</h3>
            <p style="color: #cbd5e1;"><strong>Date:</strong> {next_game['date']}</p>
            <p style="color: #cbd5e1;"><strong>Location:</strong> {'Home' if next_game['is_home'] else 'Away'}</p>
            <p style="color: #cbd5e1;"><strong>Days Rest:</strong> {next_game['days_rest']} {'(Back-to-back)' if next_game['days_rest'] == 0 else ''}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Prediction results card
        st.markdown(f"""
        <div class="prediction-card">
            <h2 style="color: #e2e8f0; margin-top: 0; text-align: center; margin-bottom: 2rem;">Predicted Performance - {st.session_state.player_name}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Predictions in styled columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="prediction-metric">
                <div class="prediction-metric-label">Points</div>
                <div class="prediction-metric-value">{prediction['points']:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="prediction-metric">
                <div class="prediction-metric-label">Rebounds</div>
                <div class="prediction-metric-value">{prediction['rebounds']:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="prediction-metric">
                <div class="prediction-metric-label">Assists</div>
                <div class="prediction-metric-value">{prediction['assists']:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Key factors
        with st.expander("📈 Key Factors"):
            features = prediction['features']
            st.write("**Recent Form:**")
            st.write(f"- Last 3 games: {features.get('pts_last_3', 0):.1f} pts")
            st.write(f"- Last 5 games: {features.get('pts_last_5', 0):.1f} pts")
            st.write(f"- Season average: {features.get('pts_season_avg', 0):.1f} pts")
            
            st.write("**Trend:**")
            trend = features.get('pts_trend', 0)
            trend_arrow = "📈" if trend > 0 else "📉" if trend < 0 else "➡️"
            st.write(f"- {trend_arrow} {trend:+.2f} pts/game (last 5 games)")
            
            st.write("**Matchup History:**")
            st.write(f"- vs {next_game['opponent']}: {features.get('pts_vs_opp_avg', 0):.1f} pts (avg)")
        
        # Download training report if available
        if st.session_state.training_report is not None:
            st.markdown("---")
            st.subheader("📊 Model Training Report")
            st.download_button(
                label="📥 Download Training Report",
                data=st.session_state.training_report,
                file_name=f"{st.session_state.player_name.replace(' ', '_')}_training_report.txt",
                mime="text/plain",
                use_container_width=True
            )
            with st.expander("📄 Preview Training Report"):
                st.text(st.session_state.training_report)
        
        # Multiple upcoming games
        st.subheader("📅 Upcoming Games Predictions")
        
        try:
            upcoming_games = get_all_upcoming_games(
                st.session_state.player_name,
                st.session_state.player_data,
                num_games=5
            )
            
            if upcoming_games and st.session_state.predictor is not None:
                predictions_list = []
                for game in upcoming_games:
                    pred = st.session_state.predictor.predict_game(
                        opponent=game['opponent'],
                        is_home=game['is_home'],
                        days_rest=game['days_rest']
                    )
                    predictions_list.append({
                        'Date': game['date'],
                        'Opponent': f"{'vs' if game['is_home'] else '@'} {game['opponent']}",
                        'Location': 'Home' if game['is_home'] else 'Away',
                        'Days Rest': game['days_rest'],
                        'Predicted PTS': f"{pred['points']:.1f}",
                        'Predicted REB': f"{pred['rebounds']:.1f}",
                        'Predicted AST': f"{pred['assists']:.1f}"
                    })
                
                upcoming_df = pd.DataFrame(predictions_list)
                st.dataframe(upcoming_df, use_container_width=True)
            else:
                st.info("No upcoming games found in the next 30 days")
        
        except Exception as e:
            st.warning(f"Could not fetch multiple upcoming games: {str(e)}")


if __name__ == "__main__":
    main()

