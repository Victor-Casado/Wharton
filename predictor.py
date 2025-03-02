import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import warnings
import joblib
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Load datasets
games = pd.read_csv("games_2022.csv")
east_games = pd.read_csv("East Regional Games to predict.csv")

# Validate that each game has exactly 2 rows
game_counts = games['game_id'].value_counts()
if not all(game_counts == 2):
    raise ValueError("Not every game_id has exactly 2 rows!")

# Split into home and away games based on the 'home_away' column
home_games = games[games['home_away'].str.lower() == 'home'].copy()
away_games = games[games['home_away'].str.lower() == 'away'].copy()

if len(home_games) != len(away_games):
    raise ValueError("Mismatch between number of home and away rows!")

# Merge home and away rows on game_id using suffixes to distinguish columns
merged = pd.merge(home_games, away_games, on='game_id', suffixes=('_home', '_away'))

# Compute the winner using team scores before dropping these columns
merged['winner'] = np.where(
    merged['team_score_home'] > merged['team_score_away'],
    merged['team_home'],
    merged['team_away']
)

# Build the merged DataFrame with the desired columns (including the computed winner)
games_merged = pd.DataFrame({
    'game_id': merged['game_id'],
    'description': merged['game_date_home'],  # using home game's date as description
    'team_home': merged['team_home'],
    'team_away': merged['team_away'],
    'home_away_NS': merged['home_away_NS_home'],  # assume both rows agree on neutral site info
    'rest_days_Home': merged['rest_days_home'],
    'rest_days_Away': merged['rest_days_away'],
    'travel_dist_Home': merged['travel_dist_home'],
    'travel_dist_Away': merged['travel_dist_away'],
    'winner': merged['winner']
})

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


# Save the trained model and encoder to disk
joblib.dump(model, 'model.pkl')
joblib.dump(ohe, 'encoder.pkl')
