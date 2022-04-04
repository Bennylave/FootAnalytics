import pandas as pd
import numpy as np
import json
from typing import Union, List

cols = ['id',
        'period',
        'minute',
        'second',
        'possession',
        'duration',
        'type_name',
        'possession_team_name',
        'play_pattern_name',
        'team_name_player',
        'location_x',
        'location_y',
        'player_name',
        'position_name']


def search_competition(competition_name, season):
    """

    :param competition_name:
    :param season:
    :return: IDs for both competition and the season
    """
    # Reading comptetitions .json
    df = pd.read_json("data/competitions.json")

    # Getting competition
    competition = df[(df['competition_name'] == str(competition_name)) & (df['season_name'] == str(season))]
    # Getting competition_id
    competition_id = str(competition.iloc[0]['competition_id'])
    # Getting season_id
    season_id = str(competition.iloc[0]['season_id'])

    return competition_id, season_id


def search_match(competition_id, season_id, home, away):
    """

    :param competition_id:
    :param season_id:
    :param home: name of home team
    :param away: name of away team
    :return: Match ID given the parameters
    """
    df = pd.read_json('data/matches/' + str(competition_id) + '/' + str(season_id) + '.json')

    df[['home_team_id', 'home_team_name',
        'home_team_gender', 'home_team_group',
        'home_team_manager_country', 'home_team_managers']] = \
        df['home_team'].apply(pd.Series)

    df[['away_team_id', 'away_team_name',
        'away_team_gender', 'away_team_group',
        'away_team_manager_country', 'away_team_managers']] = \
        df['away_team'].apply(pd.Series)

    match = df[
        (df['home_team_name'] == str(home)) & (df['away_team_name'] == str(away))]
    match_id = str(match.iloc[0]['match_id'])
    return match_id


def read_lineups(match_id):
    """

    :param match_id:
    :return: teams lineups given the match id
    """
    lineups = pd.read_json('data/lineups/' + match_id + '.json')
    lineups = lineups.explode("lineup").reset_index(drop=True).copy()
    lineups[["player_id", "full_name", "nickname", "jersey_number", "country"]] = lineups["lineup"].apply(
        pd.Series).copy()

    return lineups


def read_events(match_id):
    """

    :param match_id:
    :return: df with relevant events given the match id
    """
    # Readin match events json
    f = open("data/events/" + str(match_id) + ".json")
    data = json.load(f)
    # Noramlizing the json
    events = pd.json_normalize(data, sep="_")
    # Separating x and y into different columns
    events[["location_x", "location_y"]] = events['location'].apply(pd.Series)
    # Getting events which player specific
    events_players = events.copy()[~events.player_name.isnull()]

    # reading lineups and joining with events_df
    lineups = read_lineups(match_id)
    events_players = events_players.join(lineups.set_index("player_id"), on="player_id", how='inner', rsuffix="_player")
    # Ignoring specified columns
    events_players = events_players.drop(["team_id", "team_id_player", "team_name", "lineup", "full_name"], axis=1)
    # Replacing player name with nickname where exists
    events_players["player_name"] = np.where(~events_players['nickname'].isnull(), events_players['nickname'],
                                             events_players["player_name"])
    # Separating country name and id into different columns
    events_players[["country_id", "country_name"]] = events_players['country'].apply(pd.Series)

    return events, events_players


def read_passes(df: pd.DataFrame):
    """

    :param df: events df that has passes
    :return: df with passes only
    """
    # Ignoring passes with unknown outcome or injury clearances
    data_passes = df[(df["type_name"] == "Pass") & (~df["pass_outcome_name"].isin(["Unknown", "Injury Clearance"]))] \
        .copy()

    data_passes[["location_x", "location_y"]] = data_passes['location'].apply(pd.Series)

    # Subsetting events columns to thos including pass related info
    passes_cols = [x for x in data_passes.columns.tolist() if ((x.startswith("pass")) & (not x.endswith("_id")))]
    # Include pass related info in list of columns to keep
    if not all(item in cols for item in passes_cols):
        cols.extend(passes_cols)
    # Subset passes df to include only pass related info
    passes = data_passes[cols]

    return passes



def convert_xy_locations(x: list = None, y: list = None, is_shot=False):
    """
    :param x: list of coordinates x
    :param y: list of coordinates y
    :param is_shot: boolean if it is shot or not
    :return: converted coordinates to (105,68) system
    """
    new_x = [i * 105 / 120 for i in x]
    if is_shot:
        new_y = [abs((i * 68 / 80) - 68) for i in y]
    else:
        new_y = [i * 68 / 80 for i in y]
    return new_x, new_y



def read_passes_end_location(df: pd.DataFrame, passes_index_list: list):
    """

    :param df: df with passes
    :param passes_index_list: index of passes in the df
    :return: passes end location coordinates
    """
    # List with passes end location indexes
    passes_end_location_index = [x + 1 for x in passes_index_list]
    passes_end_location = df[df.index.isin(passes_end_location_index)].sort_index(ascending=True)

    # Getting location x of ball receipt
    passes_end_location["location_x"] = np.where(
        passes_end_location['possession_team_name'] != passes_end_location['team_name_player'],
        120 - passes_end_location["location_x"],
        passes_end_location["location_x"])

    # Getting location y of ball receipt
    passes_end_location["location_y"] = np.where(
        passes_end_location['possession_team_name'] != passes_end_location['team_name_player'],
        80 - passes_end_location["location_y"],
        passes_end_location["location_y"])

    return passes_end_location


def split_by_event_chars(df: pd.DataFrame, char: list or str):
    """

    :param df: df with events
    :param char: characteristics of event that are of interest
    :return: dict of dfs grouped by the characteristics of interest
    """
    df_dict = {k: v for k, v in df.groupby(char, dropna=False)}

    if isinstance(char, List):
        df_dict = {'_'.join(k).replace(" ", ""): v for k, v in df_dict.items()}

    return df_dict
