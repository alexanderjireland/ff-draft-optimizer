# py file for obtaining ADP data from fantasy pros

# Import statements
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from io import StringIO

def retreive_adp_data(year):
    URL = f'https://www.fantasypros.com/nfl/adp/ppr-overall.php?year={year}'
    response = requests.get(URL)
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', id='data')
    return pd.read_html(StringIO(str(table)))[0]

def adp_data_all_years(most_recent_year, save_to_csv=False, csv_filepath=None):
    adp_dfs = []
    for year in range(2015, most_recent_year+1):
        print(f'Retreiving ADP data from {year}...')
        df = retreive_adp_data(year)
        df['year'] = year
        adp_dfs.append(df)
    df = pd.concat(adp_dfs)
    if save_to_csv:
        df.to_csv(csv_filepath, index=False)
        print(f'Saved data to {csv_filepath}')
    return df