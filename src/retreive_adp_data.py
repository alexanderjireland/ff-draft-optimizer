# py file for obtaining ADP data from fantasy pros

# Import statements
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from io import StringIO
import re

def retreive_adp_data(year):
    URL = f'https://www.fantasypros.com/nfl/adp/ppr-overall.php?year={year}'
    response = requests.get(URL)
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', id='data')
    return pd.read_html(StringIO(str(table)))[0]

def parse_player_info(info):
    if not isinstance(info, str) or not info.strip():
        return [None, None, None]

    bye_match = re.search(r'\((\d*)\)\s*$', info) # Look for digits in parenthesis
    bye = bye_match.group(1) if bye_match else None

    name_team_str = re.sub(r'\s*\(\d*\)\s*$', '', info).strip()

    parts = name_team_str.split()

    suffixes = {'Jr.', 'Sr.', 'Jr', 'Sr', 'II', 'III', 'IV', 'V', 'VI', 'VII'}
    if parts and re.fullmatch(r'[A-Z]{2,3}', parts[-1]) and parts[-1] not in suffixes:
        team = parts[-1]
        name_parts = parts[:-1]
    else:
        team = None
        name_parts = parts

    name = ' '.join(name_parts)

    if bye == '':
        bye = None

    return [name, team, bye]

def split_adp_data(adp_df):
    # Splits POS column into position and position length, also splits Player Team (Bye) into full_name, team, and bye_week.
    adp_df = adp_df[~adp_df['POS'].str.startswith('DST')] # Removes defenses from df
    adp_df[['position', 'position_rank']] = adp_df['POS'].str.extract(r'([A-Za-z]+)(\d+)')
    adp_df[['full_name', 'team', 'bye_week']] = adp_df['Player Team (Bye)'].apply(parse_player_info).apply(pd.Series)
    return adp_df

def adp_data_all_years(most_recent_year, split=False, save_to_csv=False, csv_filepath=None):
    adp_dfs = []
    for year in range(2015, most_recent_year+1):
        print(f'Retreiving ADP data from {year}...')
        df = retreive_adp_data(year)
        df['year'] = year
        adp_dfs.append(df)
    df = pd.concat(adp_dfs)
    if split:
        df = split_adp_data(df)
    if save_to_csv:
        df.to_csv(csv_filepath, index=False)
        print(f'Saved data to {csv_filepath}')
    return df

