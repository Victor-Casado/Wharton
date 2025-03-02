import joblib
import pandas as pd

# Load the trained model and one-hot encoder
model = joblib.load('model.pkl')
ohe = joblib.load('encoder.pkl')

# Define the feature order used in training
# The features used were: home_away_NS, rest_days_Home, rest_days_Away, travel_dist_Home, travel_dist_Away,
# followed by one-hot encoded team features for team_home and team_away.
features = ['home_away_NS', 'rest_days_Home', 'rest_days_Away', 'travel_dist_Home', 'travel_dist_Away'] \
           + list(ohe.get_feature_names_out(['team_home', 'team_away']))

# Extract model coefficients
coeffs = model.coef_[0]
intercept = model.intercept_[0]

# Get coefficients for the basic factors
coeff_home_away = coeffs[features.index('home_away_NS')]
coeff_rest_home = coeffs[features.index('rest_days_Home')]
coeff_rest_away = coeffs[features.index('rest_days_Away')]
coeff_travel_home = coeffs[features.index('travel_dist_Home')]
coeff_travel_away = coeffs[features.index('travel_dist_Away')]

# Extract team coefficients (they come after the first 5 features)
team_features = features[5:]
team_home_coeff = {}
team_away_coeff = {}

for feat, coef in zip(team_features, coeffs[5:]):
    # Expect features of the form "team_home_TeamName" or "team_away_TeamName"
    if feat.startswith("team_home_"):
        team_name = feat[len("team_home_"):]
        team_home_coeff[team_name] = coef
    elif feat.startswith("team_away_"):
        team_name = feat[len("team_away_"):]
        team_away_coeff[team_name] = coef

# Compute a net team strength for each team as (home coefficient - away coefficient)
team_strength = {}
all_teams = set(team_home_coeff.keys()) | set(team_away_coeff.keys())
for team in all_teams:
    home_val = team_home_coeff.get(team, 0)
    away_val = team_away_coeff.get(team, 0)
    team_strength[team] = home_val - away_val

# Sort the team strengths in descending order (highest strength first)
sorted_team_strength = sorted(team_strength.items(), key=lambda x: x[1], reverse=True)

# Print sorted team strengths
print("Team Strengths (in terms of home vs. away coefficients) sorted:")
for team, strength in sorted_team_strength:
    print(f"{team}: {strength:.4f}")

# Build and print the general equation.
# For a game where Team1 is the home team and Team2 is the away team,
# let:
#   team1_strength = team_strength[Team1]
#   team2_strength = team_strength[Team2]
#
# Then, the log-odds for a win for the home team is given by:
#
# log_odds = intercept
#          + (team1_strength - team2_strength)
#          + (coeff_home_away * home_away_NS)       [typically, home_away_NS is 1 for home advantage]
#          + (coeff_rest_home * rest_days_home - coeff_rest_away * rest_days_away)
#          + (coeff_travel_away * travel_dist_away - coeff_travel_home * travel_dist_home)
#
# Which can be interpreted as:
#     (Team1_strength - Team2_strength) + home + (rest1 - rest2) + (travel2 - travel1) + intercept

print("\nGeneral Equation for Predicting Home Team Win Log-Odds:")
print(f"log_odds = {intercept:.4f} + (Team1_strength - Team2_strength) "
      f"+ {coeff_home_away:.4f} * home_away_NS "
      f"+ {coeff_rest_home:.4f} * rest_days_home - {coeff_rest_away:.4f} * rest_days_away "
      f"+ {coeff_travel_away:.4f} * travel_dist_away - {coeff_travel_home:.4f} * travel_dist_home")

print("\nInterpreting the Equation:")
print(" - (Team1_strength - Team2_strength) captures the inherent difference in team strength between the home and away teams.")
print(" - The term with home_away_NS represents the home advantage (typically 1 if the game is at home).")
print(" - The rest days and travel distance terms reflect the impact of recovery and travel burden differences between teams.")
print("Thus, the overall formula is of the form:")
print("     (team1 - team2) + home + (rest1 - rest2) + (travel2 - travel1) + intercept")
