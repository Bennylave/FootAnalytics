import numpy as np
import pandas as pd
from scipy.stats import boxcox
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler


def split_train_test(df: pd.DataFrame, test_ids:list):
    train = df.copy()[~df.index.isin(test_ids)]
    test = df.copy()[df.index.isin(test_ids)]
    return train, test


def split_feature_target(train:pd.DataFrame, test:pd.DataFrame,target_col:str):
    X_train = train.copy().drop(target_col, axis=1)
    y_train = train.copy()[[target_col]]

    X_test = test.copy().drop(target_col, axis=1)
    y_test = test.copy()[[target_col]]

    return X_train, y_train, X_test, y_test


def mlp_fit_predict(mlp:MLPClassifier, X_train:pd.DataFrame, y_train:pd.DataFrame, X_test:pd.DataFrame):
    mlp.fit(X_train, np.ravel(y_train))
    predictions = mlp.predict_proba(X_test)[:,1]

    return predictions


def preprocess_shots_data(data:pd.DataFrame):
    data_ = data.copy()

    data_["duration"] = np.sqrt(data_["duration"])

    data_["location_x"] = boxcox(data_["location_x"])[0]

    data_ = data_.drop(["minute", "second"], axis=1)

    data_["outcome"] = data_["outcome"].apply(lambda x: 1.0 if x == "Goal" else 0.0)

    min_max = MinMaxScaler()
    scaled_cols = ['possession', 'duration', 'location_x', 'location_y']
    data_[scaled_cols] = min_max.fit_transform(data_[scaled_cols])

    encoded_cols = ['under_pressure', 'play', 'type', 'technique',
                    'body_part', 'first_time', 'one_on_one', 'aerial_won',
                    'pos', 'redirect', 'deflected', 'open_goal', 'follows_dribble']

    df_encoded_cols = pd.get_dummies(data_[encoded_cols], drop_first=True)
    data_ = data_.drop(encoded_cols, axis=1)
    data_ = pd.concat([data_, df_encoded_cols], axis=1)

    return data_