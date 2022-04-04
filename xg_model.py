import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
import model_utils as mu

# Constant used for the model
seasons = {"2004/2005": "37", "2005/2006": "38", "2006/2007": "39", "2007/2008": "40", "2008/2009": "41",
           "2009/2010": "21", "2010/2011": "22", "2011/2012": "23", "2012/2013": "24", "2013/2014": "25",
           "2014/2015": "26", "2015/2016": "27", "2016/2017": "2", "2017/2018": "1", "2018/2019": "4",
           "2019/2020": "42"}

# Using LaLiga competition
competition_id = "11"

# Reading model data
data = pd.read_csv('data/clean/xg/model_data.csv', index_col="id")

# Preprocessing the shots data using model_utils
data = mu.preprocess_shots_data(data)

# training a model for each season
for season, season_id in seasons.items():
    # Reading matches id's
    full_path = 'data/matches/' + competition_id + "/" + season_id + '.json'
    match_df = pd.read_json(full_path)
    matches_id = match_df["match_id"].tolist()

    events_path = "data/events/"
    shot_ids = list()
    # Reading events for each match
    for match in matches_id:
        event_df = pd.read_json(events_path + str(match) + ".json")
        match_shots = event_df.dropna(subset=["shot"], axis=0)["id"].tolist()
        shot_ids.extend(match_shots)

    # Splitting data into train and test
    train, test = mu.split_train_test(data, shot_ids)

    # Splitting X and y from train and test df
    X_train, y_train, X_test, y_test = mu.split_feature_target(train, test, target_col="outcome")

    # Instantiating an MLP using defined architecture and hyperparameters (see notebook "xg_model_selection.ipynb")
    mlp = MLPClassifier(max_iter=1000, activation='relu', hidden_layer_sizes=(32, 16, 8, 4), learning_rate='constant',
                        solver='adam')

    # Fitting the model
    mlp.fit(X_train, np.ravel(y_train))

    # Making predictions
    test["pred"] = mlp.predict_proba(X_test)[:, 1]

    predictions = test[["pred"]]

    # Storing predictions as .csv
    predictions.to_csv('data/clean/xg/predictions_' + season.replace("/", "_") + '.csv')
