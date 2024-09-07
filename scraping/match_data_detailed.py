import json
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from get_detailed_match_data import get_detailed_nrl_data
import ENVIRONMENT_VARIABLES as EV

variables = [
    "Year", "Win", "Defense", "Attack", "Margin", "Home", "Versus", "Round"
]

select_year = "2024"
select_round = 2

years = [select_year]

# Initialize an empty dictionary to store data for each year
match_data = {}
year = select_year

with open(f"../data/nrl_data_all_years.json", 'r') as file:
    data = json.load(file)
    match_data = data['NRL']

# Create a DataFrame with columns representing combinations of team and variable names
df = pd.DataFrame(
    columns=[f"{team} {variable}" for team in EV.TEAMS for variable in variables]
)

def get_game_data(round, game):
    h_team: str = game['Home']
    a_team: str = game['Away']
    game_key: str = f"{h_team} v {a_team}"
    print(f"Collecting data for game {game_key}")
    
    try:
        game_data = get_detailed_nrl_data(
            round=round,
            year=year,
            home_team=h_team.lower(),
            away_team=a_team.lower()
        )
        return {game_key: game_data}
    except Exception as ex:
        print(f"Error collecting data for {game_key}: {ex}")
        return None

# Iterate over each round
match_json_datas = []
for round in range(1, select_round):
    print(f"Collecting detailed data for round: {round}")
    try:
        # Extract data for the current round
        round_data = match_data[year][f'Round {round}']
        
        # Use ThreadPoolExecutor to collect game data concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_game = {executor.submit(get_game_data, round, game): game for game in round_data}
            round_data_scores = []
            
            for future in as_completed(future_to_game):
                result = future.result()
                if result:
                    round_data_scores.append(result)
        
        match_json_datas.append({round: round_data_scores})

    except Exception as ex:
        print(f"Error processing round {round}: {ex}")

overall_data = {
    "NRL": match_json_datas
}
overall_data_json = json.dumps(overall_data, indent=4)

# Write JSON data to a file
with open(f"../data/nrl_detailed_match_data_{select_year}.json", "w") as file:
    file.write(overall_data_json)