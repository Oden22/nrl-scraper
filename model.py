import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import joblib
import os

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
                            'home_all_run_metres': int(match['home']['all_run_metres'].replace(',', '')),
                            'away_all_run_metres': int(match['away']['all_run_metres'].replace(',', '')),
                            'home_post_contact_metres': int(match['home']['post_contact_metres'].replace(',', '')),
                            'away_post_contact_metres': int(match['away']['post_contact_metres'].replace(',', '')),
                            'home_tackle_breaks': int(match['home']['tackle_breaks']),
                            'away_tackle_breaks': int(match['away']['tackle_breaks']),
                            'home_offloads': int(match['home']['offloads']),
                            'away_offloads': int(match['away']['offloads']),
                            'home_kicks': int(match['home']['kicks']),
                            'away_kicks': int(match['away']['kicks']),
                            'home_missed_tackles': int(match['home']['missed_tackles']),
                            'away_missed_tackles': int(match['away']['missed_tackles']),
                            'result': result
                        }
                        matches.append(match_data)
    
    return pd.DataFrame(matches)

# Enhanced Feature Engineering
def enhanced_engineer_features(df):
    features = ['score', 'possession', 'completion_rate', 'tackle_efficiency', 'line_breaks', 'errors',
                'all_run_metres', 'post_contact_metres', 'tackle_breaks', 'offloads', 'kicks', 'missed_tackles']
    
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
    
    # Add win streak feature
    df['home_win_streak'] = df.groupby('home_team')['result'].rolling(window=5, min_periods=1).sum().reset_index(level=0, drop=True)
    df['away_win_streak'] = df.groupby('away_team')['result'].rolling(window=5, min_periods=1).apply(lambda x: 5 - x.sum()).reset_index(level=0, drop=True)
    
    # Add head-to-head feature
    def get_head_to_head(row):
        h2h = df[((df['home_team'] == row['home_team']) & (df['away_team'] == row['away_team'])) |
                 ((df['home_team'] == row['away_team']) & (df['away_team'] == row['home_team']))]
        if len(h2h) > 0:
            return h2h['result'].mean()
        return 0.5  # If no previous matches, return 0.5
    
    df['head_to_head'] = df.apply(get_head_to_head, axis=1)
    
    return df

# Prepare features for model
def prepare_features(df):
    features = ['score_avg', 'possession_avg', 'completion_rate_avg', 'tackle_efficiency_avg', 'line_breaks_avg', 'errors_avg',
                'all_run_metres_avg', 'post_contact_metres_avg', 'tackle_breaks_avg', 'offloads_avg', 'kicks_avg', 'missed_tackles_avg']
    X = []
    y = []
    team_names = []
    
    for _, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        
        home_features = [row[f'{home_team}_{feature}'] for feature in features]
        away_features = [row[f'{away_team}_{feature}'] for feature in features]
        
        additional_features = [row['home_win_streak'], row['away_win_streak'], row['head_to_head']]
        
        X.append(home_features + away_features + additional_features)
        y.append(row['result'])
        team_names.append((home_team, away_team))
    
    return np.array(X), np.array(y), team_names

# Model Selection and Hyperparameter Tuning
def select_and_tune_model(X_train, y_train):
    models = {
        'random_forest': RandomForestClassifier(),
        'gradient_boosting': GradientBoostingClassifier(),
        'svm': SVC(probability=True)
    }

    param_grids = {
        'random_forest': {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20, 30]},
        'gradient_boosting': {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.3]},
        'svm': {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
    }

    best_model = None
    best_score = 0

    for name, model in models.items():
        grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_
    
    return best_model, best_score

# Handle Class Imbalance
def handle_class_imbalance(X, y):
    # SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Class Weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = dict(zip(np.unique(y), class_weights))
    
    return X_resampled, y_resampled, class_weight_dict

# Evaluate Model
def evaluate_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    return scores.mean(), scores.std()
def analyze_feature_importance(model, feature_names):
    plt.figure(figsize=(10,6))
    plt.title("Feature Importances")
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
        indices = np.argsort(importances)[::-1]
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    else:
        plt.text(0.5, 0.5, "Feature importance not available for this model", 
                 horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    plt.show()

# Save model and related data
def save_model(model, scaler, team_latest_features, filename='nrl_model.joblib'):
    data = {
        'model': model,
        'scaler': scaler,
        'team_latest_features': team_latest_features
    }
    joblib.dump(data, filename)
    print(f"Model and related data saved to {filename}")

# Load model and related data
def load_model(filename='nrl_model.joblib'):
    if os.path.exists(filename):
        data = joblib.load(filename)
        return data['model'], data['scaler'], data['team_latest_features']
    else:
        print(f"Model file {filename} not found. Please train the model first.")
        return None, None, None

# Main execution
if __name__ == "__main__":
    # Check if a saved model exists
    model, scaler, team_latest_features = load_model()
    
    if model is None:
        # Load and prepare data
        data = json.loads(open('data/nrl_detailed_match_data_all.json').read())
        df = prepare_data(data)
        df = enhanced_engineer_features(df)

        # Prepare features for model
        X, y, team_names = prepare_features(df)

        # Handle class imbalance
        X_resampled, y_resampled, class_weight_dict = handle_class_imbalance(X, y)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Select and tune the model
        best_model, best_score = select_and_tune_model(X_train_scaled, y_train)
        print(f"Best model: {best_model}")
        print(f"Best cross-validation score: {best_score:.4f}")

        # Evaluate the model
        accuracy, std_dev = evaluate_model(best_model, X_resampled, y_resampled)
        print(f"Cross-validation accuracy: {accuracy:.4f} (+/- {std_dev:.4f})")

        # Train the final model
        best_model.fit(X_train_scaled, y_train)

        # Evaluate on test set
        y_pred = best_model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"Test set accuracy: {test_accuracy:.4f}")
        print(classification_report(y_test, y_pred))

        # Analyze feature importance
        feature_names = [f'{team}_{feature}' for team in ['home', 'away'] for feature in 
                         ['score_avg', 'possession_avg', 'completion_rate_avg', 'tackle_efficiency_avg', 'line_breaks_avg', 'errors_avg',
                          'all_run_metres_avg', 'post_contact_metres_avg', 'tackle_breaks_avg', 'offloads_avg', 'kicks_avg', 'missed_tackles_avg']] + \
                        ['home_win_streak', 'away_win_streak', 'head_to_head']
        analyze_feature_importance(best_model, feature_names)

        # Create a dictionary to store the latest features for each team
        team_latest_features = {}

        for i, (home_team, away_team) in enumerate(team_names):
            home_features = X[i][:12]  # First 12 features are home team's
            away_features = X[i][12:24]  # Next 12 features are away team's
            
            team_latest_features[home_team] = home_features
            team_latest_features[away_team] = away_features

        # Save the model
        save_model(best_model, scaler, team_latest_features)
    else:
        print("Loaded existing model.")

    # Function to predict match outcome
    def predict_match(team1, team2):
        if team1 not in team_latest_features or team2 not in team_latest_features:
            return "One or both teams not found in the dataset."
        
        home_features = team_latest_features[team1]
        away_features = team_latest_features[team2]
        
        # Note: We don't have access to the latest win streaks and head-to-head data here
        # You might want to update this part if you want to include the most recent data
        features = np.concatenate((home_features, away_features, [0, 0, 0.5])).reshape(1, -1)
        scaled_features = scaler.transform(features)
        prob = model.predict_proba(scaled_features)[0][1]
        
        return f"Probability of {team1} winning against {team2}: {prob:.2f}"

    # Interactive prediction
    while True:
        team1 = input("Enter the name of the home team (or 'quit' to exit): ")
        if team1.lower() == 'quit':
            break
        team2 = input("Enter the name of the away team: ")
        print(predict_match(team1, team2))
        print()

    # Option to retrain the model with new data
    retrain = input("Do you want to retrain the model with new data? (yes/no): ")
    if retrain.lower() == 'yes':
        # Here you would load your new data and repeat the training process
        print("Retraining process would start here. Make sure to update your data file.")
        # You can copy the training code from above and place it here