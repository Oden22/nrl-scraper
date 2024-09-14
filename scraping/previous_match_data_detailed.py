import json
from get_detailed_match_data import get_detailed_nrl_data
import concurrent.futures

variables = [
    "Year",
    "Win",
    "Defense",
    "Attack",
    "Margin",
    "Home",
    "Versus",
    "Round"
]

def get_basic_data():
    with open(f"../data/nrl_data_all_years.json", 'r') as file:
        data = json.load(file)
        return data['NRL']

def get_game_data(game, round, year):
    # Extract information about the game
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
        data = {
            game_key: game_data
        }
    except Exception as ex:
        try:
            game_data = get_detailed_nrl_data(
                round=round,
                year=year,
                home_team=h_team.lower(),
                away_team=a_team.lower())
            game_data['match']
            data = {
                f"{h_team} v {a_team}": game_data
            }
        except Exception as retry_ex:
            print(f"Retry attempt failed for {game_key}: {retry_ex}")
            # Return an empty dict or handle it as necessary when retry fails
            return {game_key: None}

    return data

import concurrent.futures

def get_round_data(match_data, round, year):
    print(f"Collecting detailed data for round: {round}")
    try:
        # Extract data for the current round
        basic_round_data = match_data[year][f'Round {round}']

        # Function to process each game
        def process_game(game):
            return get_game_data(game, round, year)
        
        # Use ThreadPoolExecutor with a max of 3 threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            round_data_scores = list(executor.map(process_game, basic_round_data))

        detailed_round_data = {round: round_data_scores}
        return detailed_round_data

    except Exception as ex:
        print(f"Error getting round: {round}: {ex}")
        return {round: []}

def get_year_data(nrl_data, year, rounds):
    print(f"Getting data for year: {year}")
    year_data = []

    for round in rounds:
        round_data = get_round_data(nrl_data, round, year)
        year_data.append(round_data)

    return year_data

def save_data(data):
    overall_data = {
        "NRL": data
    }
    overall_data_json = json.dumps(overall_data, indent=4)
    with open(f"../data/nrl_detailed_match_data_all.json", "w") as file:
        file.write(overall_data_json)

def scrape(years=["2022", "2023" ,"2024"], rounds=range(1, 27)):
    print("Loading Data")
    nrl_data = get_basic_data()
    all_years_data = {}
    for year in years:
        all_years_data[year] = get_year_data(nrl_data, year, rounds)
    
    print("Saving Data")
    save_data(all_years_data)

scrape()