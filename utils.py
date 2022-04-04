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
    df = pd.read_json("data/competitions.json")
    competition = df[(df['competition_name'] == str(competition_name)) & (df['season_name'] == str(season))]
    competition_id = str(competition.iloc[0]['competition_id'])
    season_id = str(competition.iloc[0]['season_id'])

    return competition_id, season_id


def search_match(competition_id, season_id, home, away):
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
    lineups = pd.read_json('data/lineups/' + match_id + '.json')
    lineups = lineups.explode("lineup").reset_index(drop=True).copy()
    lineups[["player_id", "full_name", "nickname", "jersey_number", "country"]] = lineups["lineup"].apply(
        pd.Series).copy()

    return lineups


def read_events(match_id):
    f = open("data/events/" + str(match_id) + ".json")
    data = json.load(f)
    events = pd.json_normalize(data, sep="_")

    events[["location_x", "location_y"]] = events['location'].apply(pd.Series)

    events_players = events.copy()[~events.player_name.isnull()]

    lineups = read_lineups(match_id)
    events_players = events_players.join(lineups.set_index("player_id"), on="player_id", how='inner', rsuffix="_player")
    events_players = events_players.drop(["team_id", "team_id_player", "team_name", "lineup", "full_name"], axis=1)
    events_players["player_name"] = np.where(~events_players['nickname'].isnull(), events_players['nickname'],
                                             events_players["player_name"])

    events_players[["country_id", "country_name"]] = events_players['country'].apply(pd.Series)

    return events, events_players


def read_passes(df: pd.DataFrame):
    data_passes = df[(df["type_name"] == "Pass") & (~df["pass_outcome_name"].isin(["Unknown", "Injury Clearance"]))] \
        .copy()

    data_passes[["location_x", "location_y"]] = data_passes['location'].apply(pd.Series)

    passes_cols = [x for x in data_passes.columns.tolist() if ((x.startswith("pass")) & (not x.endswith("_id")))]

    if not all(item in cols for item in passes_cols):
        cols.extend(passes_cols)

    passes = data_passes[cols]

    return passes


def get_event_df(df: pd.DataFrame, player_name: str, event_name: str or list, event_values=None,
                 event_values_col=None, event_subfilter_col=None, event_subfilter_value=None):
    event_dict = {}

    if isinstance(event_name, list):
        event_df = df[
            (df["player_name"].str.contains(str(player_name))) &
            (df["type_name"].isin(event_name))].copy()
    else:
        event_df = df[
            (df["player_name"].str.contains(str(player_name))) &
            (df["type_name"] == event_name)].copy()

    if event_subfilter_col is not None:
        event_df = event_df[event_df[event_subfilter_col] == event_subfilter_value]

    if isinstance(event_values, dict):
        for key, value in event_values.items():
            if value == "Not Null":
                event_dict[key] = event_df[event_df[event_values_col].notnull()]
            elif value == "Null":
                event_dict[key] = event_df[event_df[event_values_col].isnull()]
            else:
                if isinstance(value, list):
                    event_dict[key] = event_df[event_df[event_values_col].isin(value)]
                else:
                    event_dict[key] = event_df[event_df[event_values_col] == value]
    else:
        if isinstance(event_name, list):
            if len(event_name) == 1:
                event_dict[event_name[0]] = event_df.copy()
            else:
                event_dict["events"] = event_df.copy()
        else:
            event_dict[event_name.replace(" ", "_")] = event_df.copy()

    return event_dict


def convert_xy_locations(x: list = None, y: list = None, is_shot=False):
    new_x = [i * 105 / 120 for i in x]
    if is_shot:
        new_y = [abs((i * 68 / 80) - 68) for i in y]
    else:
        new_y = [i * 68 / 80 for i in y]
    return new_x, new_y


def get_event_locations(event_dict: dict, event_name: None or str, end_location=False):
    location_dict = {}

    for key, value in event_dict.items():
        location_dict["X_" + str(key)] = value.location_x.to_list()
        location_dict["Y_" + str(key)] = value.location_y.to_list()
        if end_location:
            end_locations = value[event_name.lower() + "_end_location"].to_list()
            location_dict["X_" + str(event_name) + "_End_Location"] = [i[0] for i in end_locations]
            location_dict["Y_" + str(event_name) + "_End_Location"] = [i[1] for i in end_locations]

    return location_dict


def read_passes_end_location(df: pd.DataFrame, passes_index_list: list):
    passes_end_location_index = [x + 1 for x in passes_index_list]
    passes_end_location = df[df.index.isin(passes_end_location_index)].sort_index(ascending=True)

    passes_end_location["location_x"] = np.where(
        passes_end_location['possession_team_name'] != passes_end_location['team_name_player'],
        120 - passes_end_location["location_x"],
        passes_end_location["location_x"])

    passes_end_location["location_y"] = np.where(
        passes_end_location['possession_team_name'] != passes_end_location['team_name_player'],
        80 - passes_end_location["location_y"],
        passes_end_location["location_y"])

    return passes_end_location


def split_by_event_chars(df: pd.DataFrame, char: list or str):
    df_dict = {k: v for k, v in df.groupby(char, dropna=False)}

    if isinstance(char, List):
        df_dict = {'_'.join(k).replace(" ", ""): v for k, v in df_dict.items()}

    return df_dict
