import pandas as pd
import numpy as np

# Load training dataset
games = pd.read_csv("games_2022.csv")

# Ensure each game has exactly 2 rows
game_counts = games['game_id'].value_counts()
if not (game_counts == 2).all():
    raise ValueError("Not every game_id has exactly 2 rows!")

# Separate home and away rows based on 'home_away' column
home_games = games[games['home_away'].str.lower() == 'home'].copy()
away_games = games[games['home_away'].str.lower() == 'away'].copy()

if len(home_games) != len(away_games):
    raise ValueError("Mismatch between the number of home and away rows!")

# Merge home and away rows on game_id using suffixes to distinguish columns
merged = pd.merge(home_games, away_games, on='game_id', suffixes=('_home', '_away'))

# Compute winner before dropping score columns
merged['winner'] = np.where(
    merged['team_score_home'] > merged['team_score_away'],
    merged['team_home'],
    merged['team_away']
)

# Create a list of unique teams from both home and away columns
teams = pd.concat([merged['team_home'], merged['team_away']]).unique()

# Calculate win percentage for each team
win_percentage = {}
for team in teams:
    total_games = ((merged['team_home'] == team) | (merged['team_away'] == team)).sum()
    wins = (merged['winner'] == team).sum()
    win_percentage[team] = wins / total_games * 100

# Convert to DataFrame for display and saving
win_percentage_df = pd.DataFrame(list(win_percentage.items()), columns=['Team', 'Win Percentage'])
print(win_percentage_df)

# Optionally, save to CSV
win_percentage_df.to_csv("Training_Team_Win_Percentages.csv", index=False)
