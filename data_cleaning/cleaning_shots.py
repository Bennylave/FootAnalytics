from data_cleaning import data_clean_utils
import pandas as pd
import os


total_shots = pd.DataFrame()

# For every match events dataset
for i in os.listdir("../data/events"):
    # Get its path
    full_path = os.path.join("../data/events", i)

    # Read it
    df = pd.read_json(full_path)

    # Flatten the columns location to get x and y coordinates
    df = data_clean_utils.flatten_col(df, column="location", new_cols_names=["location_x", "location_y"],
                                      drop_first=False)

    # Flatten the columns and dropping the originals:
    # type to get the event type;
    # play_pattern for the type of play;
    # possession_team;
    cols_to_flat = ["type", "play_pattern", "possession_team"]
    new_cols_name = ["event", "play", "team"]

    new_cols = zip(cols_to_flat, new_cols_name)

    for col_to_flat, new_col_name in new_cols:
        df = data_clean_utils.flatten_col(df, column=col_to_flat, new_cols_names=[new_col_name])

    # Filter the original df to include shots only.
    # Index is reset to ensure that upon concatenating shots and normalized shots the correct indexes are being concatenated
    shots = df[df["event"] == "Shot"].copy().reset_index(drop=True)

    # Flatten deeply nested column shot to fetch every information related to the shot itself like its outcome, which body part was used etc.
    df_normalized_shots = pd.json_normalize(shots["shot"])

    # Concatenating events related to the shot with the remainder of the row
    shots = pd.concat([shots,df_normalized_shots], axis=1)

    # Flatten the position and player columns
    # The columns position and player may have NaNs because some events are unrelated to any player in particular.
    # Now that we have the shots only info, every event is associated to a player and position.
    shots = data_clean_utils.flatten_col(shots, column='position', new_cols_names=["pos"])
    shots = data_clean_utils.flatten_col(shots, column='player', new_cols_names=["name"])

    # Renaming the columns to indexable names
    shots = data_clean_utils.rename_cols(shots, replace=".name", by="")

    # Dropping unnecessary columns for this particular sort of event
    shots = shots.drop(['index', 'period', 'timestamp',
                        'tactics', 'related_events', 'pass',
                        'carry', 'out', 'counterpress', 'ball_receipt',
                        'clearance', 'duel', 'foul_committed', 'dribble',
                        'goalkeeper', 'shot', 'interception', '50_50', 'block',
                        'ball_recovery', 'miscontrol', 'foul_won', 'off_camera',
                        'substitution', 'statsbomb_xg', 'end_location', 'freeze_frame',
                        'technique.id', 'body_part.id', 'type.id', 'outcome.id', 'injury_stoppage',
                        'bad_behaviour', 'location', 'key_pass_id','half_start','kick_off',
                        'player_off','half_end'], axis=1, errors='ignore')


    # Imputing NaNs. In columns related to shot events, NaNs appear due to JSON fields only existing when a certain
    # characteristic of the shot is True such as one_on_one or deflected. If NaN, then impute with False or 0.
    shots = data_clean_utils.impute_nan(shots)

    # Exporting match shots data to csv named after the match_id
    shots.to_csv("../data/clean/shots/" + i.split(".")[0] + ".csv", index=False)

    total_shots = total_shots.append(shots)


total_shots.set_index("id")

total_shots = data_clean_utils.impute_nan(total_shots)

model_data = total_shots.drop(['team','event','name','saved_to_post','saved_off_target'], axis=1)

model_data.to_csv('../data/clean/xg/model_data.csv')