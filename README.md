# NBA Player Performance Prediction

A Docker-powered machine learning system for predicting NBA player performance with a web interface.

> 🚀 **Get started in 3 commands** - See [Quick Start](#quick-start-with-docker) below

## Features

- 📊 **Data Fetching**: Retrieve player game logs from NBA API
- 🔧 **Feature Engineering**: Create ML-ready features from raw game data
- 🤖 **ML Models**: Train models to predict points, rebounds, and assists
- 🔮 **Predictions**: Predict performance for upcoming games
- 🌐 **Web Interface**: User-friendly Streamlit web application
- 🐳 **Docker Support**: Run everything in a containerized environment

## Quick Start with Docker

**Requirements:**
- Docker & Docker Compose installed ([get Docker here](https://docs.docker.com/get-docker/))
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
- ✅ Select any NBA player and season
- ✅ View historical game performance data
- ✅ Automatically engineer ML features
- ✅ Predict next game performance
- ✅ Export data as CSV

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

