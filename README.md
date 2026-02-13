# NBA Player Performance Prediction

A Docker-powered machine learning system for predicting NBA player performance with a web interface.

> 🚀 **Get started in 3 commands with docker** - See below


<img width="1400" height="600" alt="Screenshot1" src="https://github.com/user-attachments/assets/724ccf8e-7567-4417-a547-19dce7f02426" />




<img width="1400" height="600" alt="Screenshot3" src="https://github.com/user-attachments/assets/2872fa26-cda7-495c-b74a-28c60fb8c6f0" />


## Features

- **Data Fetching**: Retrieve player game logs from NBA API
- **Feature Engineering**: Create ML-ready features from raw game data
- **ML Models**: Train models to predict points, rebounds, and assists
- **Predictions**: Predict performance for upcoming games
- **Web Interface**: User-friendly Streamlit web application
- **Docker Support**: Run everything in a containerized environment

## Quick Start with Docker

**Requirements:**
- Docker
- ~500MB free disk space
- Internet connection

**Start the app:**

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/NBA-stats-ML-estimator.git
cd NBA-stats-ML-estimator

# 2. Start the Docker container
docker-compose up

# 3. Open in your browser
# → http://localhost:8501
```

**That's it!** The first run will take 1-2 minutes to build the Docker image. Subsequent runs are instant.

**To stop the app:**
```bash
docker-compose down
```

---

## Features Overview

Once running, the web app lets you:
-  Select any NBA player and seasons of your choice for the data collection
-  View historical game performance data
-  Automatically create features for the ML model
-  Run model and predict next game performance
-  Export data as CSV

---

## Project Structure

```
NBA-stats-ML-estimator/
├── web/                    # Streamlit web application
│   ├── app.py              # Main web app
│   └── README.md           # App documentation
├── scripts/                # ML pipeline
│   ├── fetch_nba_data.py   # NBA API data fetching
│   ├── feature_engineering.py
│   ├── model_train.py
│   ├── predict_next_game.py
│   └── explore_data.py
├── src/                    # Utility modules
│   ├── __init__.py
│   └── nba_utils.py
├── data/                   # Downloaded player data (CSVs)
├── models/                 # Trained ML models
├── Dockerfile              # Container definition
├── docker-compose.yml      # Docker Compose config
├── run_app.py              # App launcher
├── requirements.txt        # Python dependencies
└── README.md
```




---

## License

See LICENSE file.

