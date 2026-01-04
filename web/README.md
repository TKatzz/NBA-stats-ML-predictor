# NBA Player Performance Predictor - Web Application

A Streamlit web application for predicting NBA player performance.

## Features

- **Player Selection**: Choose any NBA player by name
- **Season Selection**: Select one or more seasons to analyze
- **Data Display**: View player game logs in an interactive table
- **CSV Export**: Download data and features as CSV files
- **Feature Engineering**: Automatically create ML-ready features
- **Next Game Prediction**: Predict performance for upcoming games
- **Upcoming Games**: View predictions for multiple upcoming games

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

From the project root directory:

```bash
streamlit run web/app.py
```

Or use the provided script:

```bash
python run_app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Usage

1. **Select Player**: Enter the full name of an NBA player (e.g., "Stephen Curry")
2. **Select Seasons**: Choose one or more seasons to include in the analysis
3. **Fetch Data**: Click "Fetch Player Data" to retrieve game logs
4. **View Data**: Browse the game log table and export if needed
5. **Create Features**: Click "Create Features" to generate ML-ready features
6. **Predict**: Click "Predict Next Game" to get predictions for upcoming games

## Project Structure

```
NBA-stats-ML-estimator/
├── web/
│   └── app.py              # Main Streamlit application
├── src/
│   └── nba_utils.py        # NBA API utility functions
├── fetch_nba_data.py       # Data fetching module
├── feature_engineering.py  # Feature creation module
├── predict_next_game.py    # Prediction module
└── requirements.txt        # Dependencies
```

## Notes

- The app uses the NBA API which has rate limits. Be patient when fetching data.
- Feature engineering may take a few moments depending on the amount of data.
- Upcoming game information is fetched from the NBA API in real-time.

