# This script fetches NRL (National Rugby League) match data for the year 2024
# and saves it to a JSON file named "nrl_data_2024.json" in the "./data"
# directory.

# Imports
from get_nrl_data import get_nrl_data
import json

def scrape(years=[2022, 2023, 2024], rounds=range(1, 27)):
    year_json = get_nrl_data(rounds, years)
    overall_data = {
        "NRL": year_json
    }

    # Convert overall data to JSON format with indentation for better
    # readability
    overall_data_json = json.dumps(overall_data, indent=4)

    with open("../data/nrl_data_all_years.json", "w") as file:
        file.write(overall_data_json)