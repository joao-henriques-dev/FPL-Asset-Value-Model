import pandas as pd

def load_training_data():
    return pd.read_csv("data/training_data/training_data.csv")

def load_fixtures_data():
    return pd.read_csv("data/25-26_season_data/fixture_difficulty_ratings.csv")

def load_goalkeeper_data():
    return pd.read_csv("data/25-26_season_data/watchlist_data/goalkeeper_data.csv")

def load_defender_data():
    return pd.read_csv("data/25-26_season_data/watchlist_data/defender_data.csv")

def load_midfielder_data():
    return pd.read_csv("data/25-26_season_data/watchlist_data/midfielder_data.csv")

def load_forward_data():
    return pd.read_csv("data/25-26_season_data/watchlist_data/forward_data.csv")