import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import LinearRegression


def kfold_validate(X_df, y_df, num_folds=8, model=LinearRegression()):
    """
    Perform k-fold cross validation and return predictions

    :param X_df:
    :param y_df:
    :param num_folds:
    :param model:
    :return:
    """
    cv = KFold(n_splits=num_folds)

    model = model

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

    :param X_df:
    :param y_df:
    :param name_df:
    :param num_folds:
    :return:
    """
    preds = kfold_validate(X_df, y_df, num_folds)

    outliers = {}

    for pred in preds:
        # acceptable range: -20 < pred < 20
        if pred > 20 or pred < -20:
            print('Found one!')
            # Get index of outlier game
            X_df.iloc[[list(preds).index(pred)]].index[0]
            # Get name fo outlier game
            game = name_df.loc[46213][1]
            outliers[game] = pred

        return outliers

