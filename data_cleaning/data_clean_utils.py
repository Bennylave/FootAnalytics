import pandas as pd
import numpy as np

def goal_or_not(df:pd.DataFrame):
    df = df.copy()
    df["Goal"] = 0
    shots = df[df["event_name"] == "Shot"]
    for index, row in shots.iterrows():
        gk = df.iloc[index + 1].goalkeeper
        try:
            goal_or_not = gk["type"]["name"]
        except:
            goal_or_not = ""
        if goal_or_not == "Goal Conceded":
            df.loc[index, 'Goal'] = 1
        else:
            pass
    return df

def flatten_name(df:pd.DataFrame, column:str, new_col:str):
    df_copy = df.copy()

    df_copy[["col_to_drop", new_col]] = pd.json_normalize(df_copy[column])
    df_copy = df_copy.drop(["col_to_drop",column], axis=1)
    return df_copy


def flatten_col(df:pd.DataFrame, column:str, new_cols_names:list, drop_first=True):
    df_copy = df.copy()
    if drop_first:
        new_cols_names.insert(0, "col_to_drop")
        df_copy[new_cols_names] = df_copy[column].apply(pd.Series)
        df_copy = df_copy.drop(["col_to_drop",column], axis=1)
    else:
        df_copy[new_cols_names] = df_copy[column].apply(pd.Series)
        df_copy = df_copy.drop([column], axis=1)
    return df_copy

def get_locations(df:pd.DataFrame):
    df_copy = df.copy()
    df_copy[["location_x", "location_y"]] = df_copy["location"].apply(pd.Series)
    return df_copy

def rename_cols(df:pd.DataFrame, replace:str, by:str):
    df_copy = df.copy()
    new_labels = [col.replace(replace,by) for col in df_copy.columns.tolist()]
    df_copy = df_copy.set_axis(new_labels, axis='columns')
    return df_copy


def impute_nan(df:pd.DataFrame):
    df_copy = df.copy()
    for i in df_copy.columns.values.tolist():
        try:
            df_copy[i] = df_copy[i].replace({True : 1.0, np.nan: 0.0})
        except:
            pass
    return df_copy
