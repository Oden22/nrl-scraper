from previous_match_data_basic import scrape as basic_scrape
from previous_match_data_detailed import scrape as detailed_scrape

years = [2022, 2023, 2024]
rounds = range(1, 27)

basic_scrape(years, rounds)
detailed_scrape(years, rounds)