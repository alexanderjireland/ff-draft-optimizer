# py file for obtaining ADP data from fantasy pros

# Import statements
import pandas as pd
import numpy as np
import requests
from rapidfuzz import fuzz, process
from bs4 import BeautifulSoup
from io import StringIO
import re

def retreive_adp_data(year):
    """
    Retrieves average draft position (ADP) data from FantasyPros for a given year.

    Parameters:
        year (int): The season year to retrieve ADP data for.

    Returns:
        pd.DataFrame: A DataFrame containing the ADP data table.
    """
    # URL for Fantasy Pros ADP data for specified year
    URL = f'https://www.fantasypros.com/nfl/adp/ppr-overall.php?year={year}'
    response = requests.get(URL)
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', id='data')

    # Return pd df of table
    return pd.read_html(StringIO(str(table)))[0]

def parse_player_info(info):
    """
    Parses a player info string to extract the full name, team abbreviation, and bye week.

    Parameters:
        info (str): A string in the format "Player Name TEAM (Bye)".

    Returns:
        list: A list containing [full_name (str or None), team (str or None), bye_week (str or None)].
    """
    if not isinstance(info, str) or not info.strip():
        return [None, None, None]
    
    # Look for bye week inside parenthesis (if it exists)
    bye_match = re.search(r'\((\d+)\)', info)
    bye = bye_match.group(1) if bye_match else None

    # Find team name before bye week
    name_team_str = re.split(r'\s*\(\d+\)', info)[0].strip()

    parts = name_team_str.split()

    # Specify what suffixes to catch in name
    suffixes = {'Jr.', 'Sr.', 'Jr', 'Sr', 'II', 'III', 'IV', 'V', 'VI', 'VII'}
    if parts and re.fullmatch(r'[A-Z]{2,3}', parts[-1]) and parts[-1] not in suffixes:
        team = parts[-1]
        name_parts = parts[:-1]
    else:
        team = None
        name_parts = parts

    name = ' '.join(name_parts)

    return [name or None, team, bye]

def split_adp_data(adp_df):
    """
    Splits and cleans ADP data into separate columns for position, position rank, full name, team, and bye week.

    Parameters:
        adp_df (pd.DataFrame): The raw ADP DataFrame.

    Returns:
        pd.DataFrame: A cleaned and structured ADP DataFrame with additional columns.
    """
    # Splits POS column into position and position length, also splits Player Team (Bye) into full_name, team, and bye_week.
    adp_df = adp_df[~adp_df['POS'].str.startswith('DST')] # Removes defenses from df
    adp_df[['position', 'position_rank']] = adp_df['POS'].str.extract(r'([A-Za-z]+)(\d+)')
    adp_df[['full_name', 'team', 'bye_week']] = adp_df['Player Team (Bye)'].apply(parse_player_info).apply(pd.Series)
    return adp_df

def adp_data_all_years(most_recent_year, split=False, save_to_csv=False, csv_filepath=None):
    """
    Retrieves and optionally processes ADP data for multiple years.

    Parameters:
        most_recent_year (int): The last year (inclusive) to retrieve data for, starting from 2015.
        split (bool): Whether to split and clean the ADP data using `split_adp_data`.
        save_to_csv (bool): Whether to save the combined DataFrame to a CSV file.
        csv_filepath (str): File path for saving the CSV, required if `save_to_csv` is True.

    Returns:
        pd.DataFrame: A combined DataFrame of ADP data across years.
    """
    # Initialize empty list to concatenate later
    adp_dfs = []
    for year in range(2015, most_recent_year+1):
        # Retrieve data from every year since 2015 up to most_recent_year
        print(f'Retreiving ADP data from {year}...')
        # Retrieve specific year ADP data
        df = retreive_adp_data(year)
        # Specify what season this ADP data comes from
        df['season'] = year
        # Append to list
        adp_dfs.append(df)
    # Concatenate all years' ADP data into one df
    df = pd.concat(adp_dfs)
    if split:
        # Split df columns through regex (highly suggested)
        df = split_adp_data(df)
    if save_to_csv:
        df.to_csv(csv_filepath, index=False)
        print(f'Saved data to {csv_filepath}')
    return df

def get_default_forbidden_pairs():
    """
    Returns a default set of name pairs that should not be matched during fuzzy matching.

    Returns:
        set: A set of frozensets, each containing two names considered easily confusable.
    """
    # Default forbidden pairs selected through trial and error, found these came up incorrectly with threshold=80
    # but wanted to keep threshold limit at 80 since it caught almost all similar names
    return {
        frozenset(['derek carrier', 'derek carr']),
        frozenset(['byron marshall', 'brandon marshall']),
        frozenset(['nick williams', 'mike williams']),
        frozenset(['kasen williams', 'karlos williams']),
        frozenset(['trey williams', 'andre williams']),
        frozenset(['ryan mallett', 'matt ryan']),
        frozenset(['nick foles', 'nick folk']),
        frozenset(['jake plummer', 'jack plummer']),
        frozenset(['bijan robinson', 'brian robinson']),
        frozenset(['jalen coker', 'jalen cropper']),
        frozenset(['deonte harris', 'deonte harty']),
        frozenset(['noah brown', 'john brown']),
        frozenset(['marquise goodwin', 'marquise brown']),
        frozenset(['dwayne washington', 'deandre washington']),
        frozenset(['trayveon williams', "ty'son williams"]),
        frozenset(['malik davis', 'mike davis']),
        frozenset(['james wright', 'jarius wright']),
    }

def get_candidates(df2, season, position, name_col, season_col, position_col):
    """
    Filters candidate names for fuzzy matching based on season and optionally position.

    Parameters:
        df2 (pd.DataFrame): The dataset to search for candidate names.
        season (int): The season year to filter on.
        position (str or None): The position to filter on (optional).
        name_col (str): Column name containing player names in df2.
        season_col (str): Column name for season in df2.
        position_col (str): Column name for position in df2.

    Returns:
        list: A list of candidate player names.
    """
    # Group by season
    group = df2[df2[season_col] == season]
    if position_col in df2.columns and position:
        # Group by position
        group = group[group[position_col] == position]
    
    # Return list of candidate names
    return group[name_col].dropna().tolist()

def find_best_match(name, candidates, forbidden_pairs, threshold):
    """
    Performs fuzzy matching to find the best candidate match for a given name.

    Parameters:
        name (str): The name to match.
        candidates (list): List of candidate names.
        forbidden_pairs (set): Set of name pairs that should not be matched.
        threshold (int): Minimum score threshold for a valid match.

    Returns:
        tuple: (best_match (str or None), score (int)), the top match and its score.
    """
    # Extract top three matches to name
    results = process.extract(name, candidates, scorer=fuzz.token_sort_ratio, limit=3)
    name_lower = name.lower()

    for match, score, _ in results:
        match_lower = match.lower()
        # Make sure the match and name pair isn't in forbidden pairs
        if frozenset([name_lower, match_lower]) in forbidden_pairs:
            continue
        # If score is above threshold, proceed, else return match:None, score:0
        if score >= threshold:
            return match, score
    return None, 0

def fuzzy_match_names_by_group(df1, df2,
                                forbidden_pairs=None,
                                name_col_df1='full_name', 
                                name_col_df2='full_name',
                                team_col='team', 
                                season_col='season',
                                position_col='position',
                                threshold=80):
    """
    Fuzzy matches player names between two DataFrames by season and team, optionally filtering by position.

    Parameters:
        df1 (pd.DataFrame): Source DataFrame to match from.
        df2 (pd.DataFrame): Target DataFrame to match against.
        forbidden_pairs (set): Optional set of forbidden name pairs to avoid.
        name_col_df1 (str): Name column in df1.
        name_col_df2 (str): Name column in df2.
        team_col (str): Team column used for grouping.
        season_col (str): Season year column.
        position_col (str): Position column to refine matching.
        threshold (int): Fuzzy match threshold (0–100).

    Returns:
        pd.DataFrame: A DataFrame of matched names with match scores.
    """
    if forbidden_pairs is None:
        # Get default forbidden pairs (carefully selected through trial and error)
        forbidden_pairs = get_default_forbidden_pairs()

    matches = []
    grouped_df1 = df1.groupby([season_col, team_col])

    # Get the best candidates to name match based on season and position
    for (season, team), group1 in grouped_df1: # Found adding team not helpful, team abbreviations not standard accross dfs, so can ignore for now
        for _, row in group1.iterrows():
            name = row[name_col_df1]
            position = row.get(position_col, None)

            # Get candidates
            candidates = get_candidates(df2, season, position, name_col_df2, season_col, position_col)

            # Find best match(es) and score(s)
            match, score = find_best_match(name, candidates, forbidden_pairs, threshold)

            # Create match df to help facilitate merge later
            matches.append({
                'team': team,
                'position': position,
                'season': season,
                'original': name,
                'matched': match,
                'score': score
            })

    return pd.DataFrame(matches)

def merge_adp_all_players(adp_data, all_players, drop_extra_cols=True):
    """
    Merges ADP data with a player dataset using fuzzy name matching and calculates fantasy performance metrics.

    Parameters:
        adp_data (pd.DataFrame): The ADP dataset.
        all_players (pd.DataFrame): A dataset of all player stats.
        drop_extra_cols (bool): Whether to drop intermediate columns after merging.

    Returns:
        pd.DataFrame: A merged DataFrame including ADP data and end-of-season ranks.
    """
    # Obtain fuzzy matched names
    matched_grouped = fuzzy_match_names_by_group(all_players, adp_data)

    # Filter out names not matched
    matched_df_clean = matched_grouped[matched_grouped['matched'].notna()]

    # Create keys to match dfs on
    all_players['merge_key'] = all_players['full_name'].str.lower()
    adp_data['merge_key'] = adp_data['full_name'].str.lower()
    matched_df_clean['original_key'] = matched_df_clean['original'].str.lower()
    matched_df_clean['matched_key'] = matched_df_clean['matched'].str.lower()

    # Merge matched_df_cleaned with all_players and adp_data
    all_players_merged = matched_df_clean.merge(all_players, left_on=['original_key', 'team', 'position', 'season'], right_on=['merge_key', 'team', 'position', 'season'], suffixes=('', '_all_players'), how='right')
    final_merged = all_players_merged.merge(adp_data, left_on=['position', 'matched_key', 'full_name', 'season'], right_on=['position', 'merge_key', 'full_name', 'season'], suffixes=('_all_players', '_adp_data'), how='left')

    # Extra columns to drop (optionally)
    if drop_extra_cols:
        final_merged = final_merged.drop(columns=['team_adp_data', 'merge_key_adp_data', 'Sleeper', 'NFL', 'RTSports', 'FFC', 'original', 'matched', 'score', 'original_key', 'matched_key', 'merge_key_all_players', 'Player Team (Bye)', 'POS', 'depth_chart_position'])
    
    # Ensure 'id' column is first col
    final_merged = final_merged[['id']+[col for col in final_merged.columns if col != 'id']]

    # Add a few rank-based columns to df
    final_merged['season_end_rank'] = final_merged.groupby(['position', 'season'])['fantasy_pts'].rank(method='dense', ascending=False).astype(int)
    final_merged['season_end_rank_diff'] = final_merged['season_end_rank'] - final_merged['Rank']
    final_merged['ESPN_season_end_rank_diff'] = final_merged['season_end_rank'] - final_merged['ESPN']

    return final_merged