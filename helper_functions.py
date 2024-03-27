import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import LinearRegression


def kfold_validate(X_df, y_df, num_folds=8):
    """
    Perform k-fold cross validation and return predictions

    :param X_df: X dataframe, usually X_train
    :param y_df: y dataframe, usually y_train
    :param num_folds: number of folds used in KFolds()

    :return: numpy array containing predictions from Linear Regression
             K-Fold Cross Validation
    """
    cv = KFold(n_splits=num_folds)

    model = LinearRegression()

    # Generate predictions from the cross validation
    preds = cross_val_predict(
        model,
        X_df,
        y_df,
        cv=cv
    )

    return preds


def find_outlier_games(X_df, y_df, name_df, num_folds=8):
    """
    Use kfold_validate() to return list of games that return
    predictions that are outliers

    :param X_df: X dataframe, usually X_train
    :param y_df: y dataframe, usually y_train
    :param name_df: dataframe containing indexed names
    :param num_folds: number of folds used in KFolds()

    :return: dictionary that contains outlier prediction,
             in the format {"Name of Game": prediction value}
    """
    preds = kfold_validate(X_df, y_df, num_folds)

    outliers = {}

    for pred in list(preds):
        # acceptable range: -20 < pred < 20
        if pred > 20.0 or pred < -20.0:
            # Get index of outlier game
            outlier_idx = X_df.iloc[[list(preds).index(pred)]].index[0]
            # Get name fo outlier game
            game = name_df.loc[outlier_idx][0]
            outliers[game] = pred

    return outliers
