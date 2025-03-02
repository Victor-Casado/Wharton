import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Load datasets
games = pd.read_csv("games_2022.csv")
east_games = pd.read_csv("East Regional Games to predict.csv")

# Merge each game into a single row (home & away in same row)
games = games.sort_values(by=['game_id', 'team'])
games_merged = games.groupby('game_id').apply(lambda g: pd.Series({
    'description': g['game_date'].iloc[0],  # Placeholder for game description
    'team_home': g['team'].iloc[0], 'team_away': g['team'].iloc[1],
    'seed_home': g['seed'].iloc[0] if 'seed' in g else np.nan,
    'seed_away': g['seed'].iloc[1] if 'seed' in g else np.nan,
    'home_away_NS': g['home_away_NS'].iloc[0],  # Home/Away neutral site
    'rest_days_Home': g['rest_days'].iloc[0], 'rest_days_Away': g['rest_days'].iloc[1],
    'travel_dist_Home': g['travel_dist'].iloc[0], 'travel_dist_Away': g['travel_dist'].iloc[1],
    'winner': g['team'].iloc[0] if g['team_score'].iloc[0] > g['team_score'].iloc[1] else g['team'].iloc[1]
})).reset_index()

# One-hot encode team names
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
team_encoded = ohe.fit_transform(games_merged[['team_home', 'team_away']])
team_encoded_df = pd.DataFrame(team_encoded, columns=ohe.get_feature_names_out(['team_home', 'team_away']))

# Merge encoded team data
games_merged = pd.concat([games_merged, team_encoded_df], axis=1)

# Define features and target
features = ['home_away_NS', 'rest_days_Home', 'rest_days_Away', 'travel_dist_Home', 'travel_dist_Away'] + list(team_encoded_df.columns)
games_merged['target'] = (games_merged['winner'] == games_merged['team_home']).astype(int)

X = games_merged[features]
y = games_merged['target']

# Handle missing values
X = X.fillna(X.mean(numeric_only=True))
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model (classification for win/loss)
model = LogisticRegression(max_iter = 5000)
#print("Features used for training:", list(X.columns))
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Wait for user confirmation to predict east regional games
input("Press Enter to predict East Regional Games...")

# Prepare East Regional games for prediction
east_games = east_games.sort_values(by=['game_id', 'team_home', 'team_away'])  # ✅ Use correct columns
east_games_merged = east_games.rename(columns={
    'rest_days_Home': 'rest_days_Home',
    'rest_days_Away': 'rest_days_Away',
    'travel_dist_Home': 'travel_dist_Home',
    'travel_dist_Away': 'travel_dist_Away',
    'home_away_NS': 'home_away_NS'
})


# One-hot encode teams in east games
east_team_encoded = ohe.transform(east_games_merged[['team_home', 'team_away']])
east_team_df = pd.DataFrame(east_team_encoded, columns=ohe.get_feature_names_out(['team_home', 'team_away']))
#print(east_games_merged)
east_games_merged = pd.concat([east_games_merged, east_team_df], axis=1)

# Define the features we want to keep (only numeric features + one-hot encoded teams)
features = list(east_team_df.columns) + ['rest_days_Home', 'rest_days_Away', 'travel_dist_Home', 'travel_dist_Away', 'home_away_NS']

# Ensure we only pass those columns
east_games_merged[features] = east_games_merged[features].fillna(east_games_merged[features].mean(numeric_only=True))

#save info
east_games_info = east_games_merged[['game_id', 'team_home', 'team_away']].copy()

# Make predictions using only the selected features
east_games_merged = east_games_merged.reindex(columns=X.columns, fill_value=0)

missing_features = set(X.columns) - set(east_games_merged.columns)
extra_features = set(east_games_merged.columns) - set(X.columns)
#print("Missing in prediction data:", missing_features)
#print("Extra in prediction data:", extra_features)


predictions_proba = model.predict_proba(east_games_merged[X.columns])


east_games_info["Team_Home_Win_Percentage"] = predictions_proba[:, 1] * 100
east_games_info["Team_Away_Win_Percentage"] = predictions_proba[:, 0] * 100

# Optionally, save the predictions
east_games_info[["game_id", "Team_Home_Win_Percentage", "Team_Away_Win_Percentage"]].to_csv("East_Regional_Predictions.csv", index=False)

print(east_games_info[["game_id", "Team_Home_Win_Percentage", "Team_Away_Win_Percentage"]])
