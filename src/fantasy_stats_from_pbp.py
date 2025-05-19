# py file to turn pbp data to season totals and calculate fantasy scores

# Import statements
import pandas as pd
import numpy as np
from tqdm import tqdm

def create_player_stats(player_id, pbp):
    """
    Generates season-level statistics for a single player based on play-by-play data.
    
    Note: A season is considered to be weeks 1-17 of the regular season, since many
    fantasy football formats start playoffs in week 14 and end in week 17. This is
    done primarily to avoid a championship game with star players sitting on the bench.

    Args:
        player_id (str): The unique ID of the player.
        pbp (pd.DataFrame): The play-by-play DataFrame containing NFL play data.

    Returns:
        pd.DataFrame: A DataFrame of the player's season totals with relevant offensive stats.
    """
    # Create two point binary column
    pbp['two_pt'] = np.where((pbp['two_point_conv_result']=='success'), 1, 0)
    # Can probably break this down to group by player_id instead of calculating individually for each player down the road
    passer = pbp[pbp['passer_player_id']==player_id][['passer_player_id', 'play_id', 'drive', 'game_id', 'week', 'season', 'passing_yards', 'pass_touchdown', 'interception', 'sack', 'two_pt']].rename(columns={'two_pt':'two_pt_pass'})
    rusher = pbp[pbp['rusher_player_id']==player_id][['rusher_player_id', 'play_id', 'drive', 'game_id', 'week', 'season', 'rush_touchdown', 'rushing_yards', 'two_pt']].rename(columns={'two_pt':'two_pt_rush'})
    receiver = pbp[pbp['receiver_player_id']==player_id][['receiver_player_id', 'play_id', 'drive', 'game_id', 'season', 'week', 'pass_touchdown', 'receiving_yards', 'two_pt']].rename(columns={'pass_touchdown':'receive_touchdown', 'two_pt':'two_pt_receive'})
    fumbler = pbp[pbp['fumbled_1_player_id']==player_id][['fumbled_1_player_id', 'play_id', 'drive', 'game_id', 'week', 'season', 'fumble_lost']]

    # Merge pass and rush data frames
    pass_rush = pd.merge(
    passer,
    rusher,
    how='outer',
    left_on=['passer_player_id', 'play_id', 'drive', 'game_id', 'week', 'season'],
    right_on=['rusher_player_id', 'play_id', 'drive', 'game_id', 'week', 'season']
    )

    # Merge pass, rush, and fumbles
    pass_rush_fumble = pd.merge(
    pass_rush,
    fumbler,
    how='outer',
    left_on=['passer_player_id', 'play_id', 'drive', 'game_id', 'week', 'season'],
    right_on=['fumbled_1_player_id', 'play_id', 'drive', 'game_id', 'week', 'season']
    )

    # Merge pass, rush, fumbles, and receptions
    all_df = pd.merge(
        pass_rush_fumble,
        receiver,
        how='outer',
        left_on=['passer_player_id', 'play_id', 'drive', 'game_id', 'week', 'season'],
        right_on=['receiver_player_id', 'play_id', 'drive', 'game_id', 'week', 'season']
    )

    # Since all two_pt completions are scored the same (whether passed, rushed, or caught), just turn all into one column
    all_df['two_pt'] = all_df[['two_pt_receive', 'two_pt_pass', 'two_pt_rush']].fillna(0).sum(axis=1)

    # Create one id column
    all_df['id'] = all_df[['passer_player_id', 'rusher_player_id', 'receiver_player_id', 'fumbled_1_player_id']].bfill(axis=1).iloc[:, 0]
    player = all_df.drop(columns=['passer_player_id', 'rusher_player_id', 'receiver_player_id', 'fumbled_1_player_id', 'two_pt_receive', 'two_pt_pass', 'two_pt_rush'])

    # filter out postseason and last few weeks of regular season (where weeks 15-17 are post-season, and anything afterward is not considered since fantasy football play has ended)
    player_filtered = player[player['week']<18]

    # obtain season total stats
    player_season = pd.DataFrame()
    cols_to_sum = ['passing_yards', 'pass_touchdown', 'rush_touchdown', 'interception', 'fumble_lost', 'rushing_yards', 'two_pt', 'receiving_yards','receive_touchdown']
    player_season['reception'] = player_filtered.groupby('season')['receiving_yards'].count()
    player_season[cols_to_sum] = player_filtered.groupby('season')[cols_to_sum].sum()
    player_season['id'] = player_id
    player_season = player_season[['id'] + [col for col in player_season.columns if col != 'id']]
    
    return player_season

def calculate_espn_ppr_score(row, ppr=True):
    """
    Calculates a fantasy football score for a given player's season row using ESPN scoring.

    Args:
        row (pd.Series): A row of player stats including yards, touchdowns, etc.
        ppr (bool, optional): If True, applies PPR (points per reception) scoring. Defaults to True.

    Returns:
        float: The fantasy points scored for that season.
    """
    if ppr:
        # PPR format:
        weights = {
            'passing_yards': .04, # 1 pt for every 25 yds
            'pass_touchdown': 4,
            'interception': -2,
            'fumble_lost': -2,
            'rushing_yards': .1,
            'rush_touchdown': 6,
            'reception': 1,
            'receive_touchdown': 6,
            'receiving_yards': .1,
            'two_pt': 2
        }
    else:
        #NonPPR format:
        weights = {
            'passing_yards': .04, # 1 pt for every 25 yds
            'pass_touchdown': 4,
            'interception': -2,
            'fumble_lost': -2,
            'rushing_yards': .1,
            'rush_touchdown': 6,
            'receive_touchdown': 6,
            'receiving_yards': .1,
            'two_pt': 2
        }
    return row[list(weights)].dot(pd.Series(weights))

def calculate_all_players_season_stats(pbp, save_to_csv=False, csv_filepath=None):
    """
    Processes all players from play-by-play data and returns season fantasy stats.

    Args:
        pbp (pd.DataFrame): The play-by-play DataFrame containing NFL play data.
        save_to_csv (bool, optional): Whether to save the final DataFrame to a CSV. Defaults to False.
        csv_filepath (str, optional): Path to save the CSV file if save_to_csv is True.

    Returns:
        pd.DataFrame: A DataFrame containing season totals and fantasy scores for all players.
    """
    # Collect all unique fantasy player ids (players that passed, rushed, or received)
    print('Collecting all unique fantasy player ids...')
    ids = pd.DataFrame()
    ids['id'] = pbp[['passer_player_id', 'rusher_player_id', 'receiver_player_id']].bfill(axis=1).iloc[:, 0]
    player_ids = ids['id'].drop_duplicates().dropna()
    print(f'Collected {len(player_ids)} player ids.')

    # Initialize empty list to carry each player's stats
    player_stats = []
    print('Now collecting data for all fantasy players...')
    for player_id in tqdm(player_ids): # Use tqdm so we know approximately how long this will take
        try:
            # Create the player's stats grouped by season
            player_season_stats = create_player_stats(player_id)
            # Calculate the player's total number of fantasy points by the end of the season
            player_season_stats['fantasy_pts'] = player_season_stats.apply(lambda row: calculate_espn_ppr_score(row), axis=1)
            # Add player's stats and fantasy totals to list
            player_stats.append(player_season_stats)
        except Exception as e:
            print(f'Error processing player id {player_id}: {e}')
    
    # Concat all players together into one df
    df = pd.concat(player_stats).reset_index(names='season')
    print('Finished collecting data from all fantasy players.')

    # Save to a .csv
    if save_to_csv:
        df.to_csv(csv_filepath, index=False)
        print(f'Saved data to {csv_filepath}')

    return df