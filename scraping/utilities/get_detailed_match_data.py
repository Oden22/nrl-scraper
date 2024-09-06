"""
Web scraper for finding NRL data related to team statistics
"""
from bs4 import BeautifulSoup
from .set_up_driver import set_up_driver

# Constants for stat categories
BARS_DATA = {
    'time_in_possession': -1, 'all_runs': -1, 'all_run_metres': -1,
    'post_contact_metres': -1, 'line_breaks': -1, 'tackle_breaks': -1,
    'average_set_distance': -1, 'kick_return_metres': -1, 'offloads': -1,
    'receipts': -1, 'total_passes': -1, 'dummy_passes': -1, 'kicks': -1,
    'kicking_metres': -1, 'forced_drop_outs': -1, 'bombs': -1, 'grubbers': -1,
    'tackles_made': -1, 'missed_tackles': -1, 'intercepts': -1,
    'ineffective_tackles': -1, 'errors': -1, 'penalties_conceded': -1,
    'ruck_infringements': -1, 'inside_10_metres': -1, 'interchanges_used': -1
}

DONUT_DATA = {
    'Completion Rate': -1, 'Average_Play_Ball_Speed': -1, 'Kick_Defusal': -1,
    'Effective_Tackle': -1
}

DONUT_DATA_2 = {
    'tries': -1, 'conversions': -1, 'penalty_goals': -1, 'sin_bins': -1,
    '1_point_field_goals': -1, '2_point_field_goals': -1, 'half_time': -1
}

DONUT_DATA_2_WORDS = [
    'TRIES', 'CONVERSIONS', 'PENALTY GOALS', 'SIN BINS', '1 POINT FIELD GOALS',
    '2 POINT FIELD GOALS', 'HALF TIME'
]


def get_detailed_nrl_data(round=1, year=2024, home_team="sea-eagles", away_team="rabbitohs"):
    """Fetch detailed NRL match data between two teams for a specific round."""
    home_team, away_team = [team.replace(" ", "-") for team in [home_team, away_team]]
    url = f"https://www.nrl.com/draw/nrl-premiership/{year}/round-{round}/{home_team}-v-{away_team}/"
    print(f"Scraping data for Round {round}: {url}")

    # Web scrape the NRL site using Selenium
    soup = get_page_source(url)

    # Extract possession data
    home_possession, away_possession = extract_possession_data(soup)

    # Extract bar chart stats
    home_bars, away_bars = extract_bar_chart_data(soup)

    # Extract donut stats
    home_donut, away_donut = extract_donut_data(soup)

    # Extract try scorers data
    home_try_data, away_try_data = extract_try_scorers(soup, home_team, away_team)

    # Determine first try scorer data
    overall_first_try_data = determine_first_try_scorer(home_try_data, away_try_data, home_team, away_team)

    # Extract top-level match stats
    home_game_stats, away_game_stats = extract_top_match_stats(soup)

    # Extract referee data
    ref_data = extract_referee_data(soup)

    # Extract ground and weather conditions
    match_conditions = extract_match_conditions(soup)

    # Combine all the data
    home_data = {**home_try_data, **home_bars, **home_donut, **home_game_stats}
    away_data = {**away_try_data, **away_bars, **away_donut, **away_game_stats}
    match_data = {**overall_first_try_data, **ref_data, **match_conditions}

    return {'match': match_data, 'home': home_data, 'away': away_data}


def get_page_source(url):
    """Get the page source for the given URL using Selenium."""
    driver = set_up_driver()
    driver.get(url)
    page_source = driver.page_source
    driver.quit()
    return BeautifulSoup(page_source, "html.parser")


def extract_possession_data(soup):
    """Extract possession data for home and away teams."""
    try:
        home_possession = soup.find('p', class_='match-centre-card-donut__value--home').text.strip()
        away_possession = soup.find('p', class_='match-centre-card-donut__value--away').text.strip()
    except Exception as e:
        print(f"Error extracting possession data: {e}")
        home_possession, away_possession = None, None
    return home_possession, away_possession


def extract_bar_chart_data(soup):
    """Extract bar chart statistics for both teams."""
    home_bars, away_bars = BARS_DATA.copy(), BARS_DATA.copy()

    try:
        home_elements = soup.find_all('dd', class_='stats-bar-chart__label--home')
        away_elements = soup.find_all('dd', class_='stats-bar-chart__label--away')

        for item, bar_name in zip(home_elements, home_bars.keys()):
            home_bars[bar_name] = item.get_text(strip=True)

        for item, bar_name in zip(away_elements, away_bars.keys()):
            away_bars[bar_name] = item.get_text(strip=True)
    except Exception as e:
        print(f"Error extracting bar chart data: {e}")
    
    return home_bars, away_bars


def extract_donut_data(soup):
    """Extract donut statistics for both teams."""
    home_donut, away_donut = DONUT_DATA.copy(), DONUT_DATA.copy()

    try:
        elements = soup.find_all("p", class_="donut-chart-stat__value")
        numbers = [element.get_text(strip=True) for element in elements]
        home_donut.update({k: v for k, v in zip(home_donut, numbers[::2])})
        away_donut.update({k: v for k, v in zip(away_donut, numbers[1::2])})
    except Exception as e:
        print(f"Error extracting donut data: {e}")
    
    return home_donut, away_donut


def extract_try_scorers(soup, home_team, away_team):
    """Extract try scorers data for both teams."""
    home_try_data = extract_team_try_data(soup, "home")
    away_try_data = extract_team_try_data(soup, "away")

    return {'team': home_team, **home_try_data}, {'team': away_team, **away_try_data}


def extract_team_try_data(soup, team):
    """Extract try scorer names and times for a given team."""
    try_data = {'try_names': [], 'try_minutes': [], 'first_try_scorer': None, 'first_try_time': None}
    
    try:
        li_elements = soup.find(f"ul", class_=f"match-centre-summary-group__list--{team}").find_all("li")
        for li in li_elements:
            text = li.get_text(strip=True).split()
            name, minute = ' '.join(text[:-1]), text[-1]
            try_data['try_names'].append(name)
            try_data['try_minutes'].append(minute)

        if try_data['try_names']:
            try_data['first_try_scorer'] = try_data['try_names'][0]
            try_data['first_try_time'] = try_data['try_minutes'][0]
    except Exception as e:
        print(f"Error extracting try scorers for {team}: {e}")
    
    return try_data


def determine_first_try_scorer(home_data, away_data, home_team, away_team):
    """Determine the first try scorer between home and away teams."""
    if home_data['first_try_time'] and (not away_data['first_try_time'] or home_data['first_try_time'] < away_data['first_try_time']):
        return {'overall_first_try_scorer': home_data['first_try_scorer'], 'overall_first_try_minute': home_data['first_try_time'], 'overall_first_try_team': home_team}
    elif away_data['first_try_time']:
        return {'overall_first_try_scorer': away_data['first_try_scorer'], 'overall_first_try_minute': away_data['first_try_time'], 'overall_first_try_team': away_team}
    return {}


def extract_top_match_stats(soup):
    """Extract top-level match statistics."""
    home_stats, away_stats = DONUT_DATA_2.copy(), DONUT_DATA_2.copy()

    try:
        span_elements = soup.find_all("span", class_="match-centre-summary-group__value")
        numbers = [span.get_text(strip=True) for span in span_elements]
        
        for key, value in zip(home_stats.keys(), numbers[::2]):
            home_stats[key] = value
        for key, value in zip(away_stats.keys(), numbers[1::2]):
            away_stats[key] = value
    except Exception as e:
        print(f"Error extracting match stats: {e}")
    
    return home_stats, away_stats


def extract_referee_data(soup):
    """Extract referee information."""
    ref_data = {'ref_names': [], 'ref_positions': [], 'main_ref': None}
    
    try:
        a_elements = soup.find_all("a", class_="match-centre-header__official")
        ref_data['ref_names'] = [a.get_text(strip=True) for a in a_elements]

        ref_pos_elements = soup.find_all("p", class_="match-centre-header__position")
        ref_data['ref_positions'] = [p.get_text(strip=True) for p in ref_pos_elements]

        if ref_data['ref_names']:
            ref_data['main_ref'] = ref_data['ref_names'][0]
    except Exception as e:
        print(f"Error extracting referee data: {e}")
    
    return ref_data


def extract_match_conditions(soup):
    """Extract match ground and weather conditions."""
    match_conditions = {'ground': None, 'weather': None}

    try:
        details_elements = soup.find_all("dd", class_="match-centre-header__detail")
        match_conditions['ground'] = details_elements[0].get_text(strip=True)
        match_conditions['weather'] = details_elements[1].get_text(strip=True)
    except Exception as e:
        print(f"Error extracting match conditions: {e}")
    
    return match_conditions