import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import json

# Data Preparation
def prepare_data(data):
    matches = []
    for year, rounds in data["NRL"].items():
        for round_data in rounds:
            for all_games in round_data.values():
                for match in all_games:
                    key = list(match.keys())[0]
                    match = match[key]
                    if match != None:
                        home_team = match['home']['team']
                        away_team = match['away']['team']
                        home_score = sum(map(int, match['home']['tries']))
                        away_score = sum(map(int, match['away']['tries']))
                        result = 1 if home_score > away_score else 0
                        
                        match_data = {
                            'home_team': home_team,
                            'away_team': away_team,
                            'home_score': home_score,
                            'away_score': away_score,
                            'home_possession': float(match['home']['possesion'].strip('%')) / 100,
                            'away_possession': float(match['away']['possesion'].strip('%')) / 100,
                            'home_completion_rate': float(match['home']['Completion Rate'].strip('%')) / 100,
                            'away_completion_rate': float(match['away']['Completion Rate'].strip('%')) / 100,
                            'home_tackle_efficiency': float(match['home']['Effective_Tackle'].strip('%')) / 100,
                            'away_tackle_efficiency': float(match['away']['Effective_Tackle'].strip('%')) / 100,
                            'home_line_breaks': int(match['home']['line_breaks']),
                            'away_line_breaks': int(match['away']['line_breaks']),
                            'home_errors': int(match['home']['errors']),
                            'away_errors': int(match['away']['errors']),
                            'result': result
                        }
                        matches.append(match_data)
    
    return pd.DataFrame(matches)

# Feature Engineering
def engineer_features(df):
    # Calculate rolling averages for the last 5 matches
    features = ['score', 'possession', 'completion_rate', 'tackle_efficiency', 'line_breaks', 'errors']
    for team in df['home_team'].unique():
        team_matches = df[(df['home_team'] == team) | (df['away_team'] == team)].sort_index()
        
        for feature in features:
            home_feature = f'home_{feature}'
            away_feature = f'away_{feature}'
            team_feature = f'{team}_{feature}_avg'
            
            team_stats = []
            for _, row in team_matches.iterrows():
                if row['home_team'] == team:
                    team_stats.append(row[home_feature])
                else:
                    team_stats.append(row[away_feature])
            
            rolling_avg = pd.Series(team_stats).rolling(window=5, min_periods=1).mean()
            df.loc[team_matches.index, team_feature] = rolling_avg.values
    
    return df

# Prepare features for model
def prepare_features(df):
    features = ['score_avg', 'possession_avg', 'completion_rate_avg', 'tackle_efficiency_avg', 'line_breaks_avg', 'errors_avg']
    X = []
    y = []
    
    for _, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        
        home_features = [row[f'{home_team}_{feature}'] for feature in features]
        away_features = [row[f'{away_team}_{feature}'] for feature in features]
        
        X.append(home_features + away_features)
        y.append(row['result'])
    
    return np.array(X), np.array(y)

data = json.loads(open('data/nrl_detailed_match_data_all.json').read())
df = prepare_data(data)
df = engineer_features(df)

# Prepare features for model
def prepare_features(df):
    features = ['score_avg', 'possession_avg', 'completion_rate_avg', 'tackle_efficiency_avg', 'line_breaks_avg', 'errors_avg']
    X = []
    y = []
    team_names = []
    
    for _, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        
        home_features = [row[f'{home_team}_{feature}'] for feature in features]
        away_features = [row[f'{away_team}_{feature}'] for feature in features]
        
        X.append(home_features + away_features)
        y.append(row['result'])
        team_names.append((home_team, away_team))
    
    return np.array(X), np.array(y), team_names

X, y, team_names = prepare_features(df)

# Split the data
X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(X, y, team_names, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# Create a dictionary to store the latest features for each team
team_latest_features = {}

for i, (home_team, away_team) in enumerate(team_names):
    home_features = X[i][:6]  # First 6 features are home team's
    away_features = X[i][6:]  # Last 6 features are away team's
    
    team_latest_features[home_team] = home_features
    team_latest_features[away_team] = away_features

# Function to predict match outcome
def predict_match(team1, team2):
    if team1 not in team_latest_features or team2 not in team_latest_features:
        return "One or both teams not found in the dataset."
    
    features = np.concatenate((team_latest_features[team1], team_latest_features[team2])).reshape(1, -1)
    scaled_features = scaler.transform(features)
    prob = model.predict_proba(scaled_features)[0][1]
    
    return f"Probability of {team1} winning against {team2}: {prob:.2f}"

# Example usage
print(predict_match("Broncos", "Dolphins"))
print(predict_match("Roosters", "Panthers"))

# Interactive prediction
while True:
    team1 = input("Enter the name of the first team (or 'quit' to exit): ")
    if team1.lower() == 'quit':
        break
    team2 = input("Enter the name of the second team: ")
    print(predict_match(team1, team2))
    print()