import concurrent.futures
from bs4 import BeautifulSoup
from set_up_driver import set_up_driver

# Helper function to fetch data for a specific round and year
def fetch_round_data(round, year):
    driver = set_up_driver()
    print(f"Fetching data for round {round}")
    url = f"https://www.nrl.com/draw/?competition=111&round={round}&season={year}"
    driver.get(url)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    match_elements = soup.find_all("div", class_="match o-rounded-box o-shadowed-box")

    matches_json = []
    for match_element in match_elements:
        match_details, match_date, home_team, home_score, away_team, away_score, venue = [
            match_element.find(html_val, class_=class_val).text.strip()
            for html_val, class_val in zip(
                ["h3", "p", "p", "div", "p", "div", "p"],
                ["u-visually-hidden", "match-header__title", "match-team__name--home",
                 "match-team__score--home", "match-team__name--away",
                 "match-team__score--away", "match-venue o-text"]
            )
        ]

        match = {
            "Details": match_details.replace("Match: ", ""),
            "Date": match_date,
            "Home": home_team,
            "Home_Score": home_score.replace("Scored", "").replace("points", "").strip(),
            "Away": away_team,
            "Away_Score": away_score.replace("Scored", "").replace("points", "").strip(),
            "Venue": venue.replace("Venue:", "").strip()
        }
        matches_json.append(match)
    print(f"Collected data for round {round}")
    driver.quit()
    return {f"Round {round}": matches_json}

# Main function to get NRL data concurrently
def get_nrl_data(rounds=[26], years=[2024]):
    driver = set_up_driver()  # Set up WebDriver once
    all_data = {}  # Store all rounds and years data

    for year in years:
        year_data = {}
        print(f"Collecting NRL data for year: {year}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Submit tasks to fetch round data concurrently
            futures = {executor.submit(fetch_round_data, round, year): round for round in rounds}

            for future in concurrent.futures.as_completed(futures):
                round = futures[future]
                try:
                    round_data = future.result()
                    year_data.update(round_data)
                except Exception as ex:
                    print(f"Error collecting data for round {round}: {ex}")

        print(f"Completed collection for year: {year}")
        all_data[f"{year}"] = year_data

    driver.quit()  # Close the WebDriver
    return all_data