import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.linear_model import LinearRegression


def kfold_validate_pred(X_df, y_df, num_folds=8):
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


def kfold_validate_score(X_df, y_df, num_folds=8, model=LinearRegression()):
    """
    Perform k-fold cross validation and returns mean RSME of CV

    :param X_df: X dataframe, usually X_train
    :param y_df: y dataframe, usually y_train
    :param num_folds: number of folds used in KFolds()
    :param model: Regression model chosen

    :return: numpy array containing predictions from Linear Regression
             K-Fold Cross Validation
    """
    scores = -1 * cross_val_score(
        model,
        X_df,
        y_df,
        scoring='neg_mean_squared_error',
        cv=KFold(n_splits=num_folds)
    )

    avg_rmse = np.mean(np.sqrt(scores))

    return avg_rmse


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
    preds = kfold_validate_pred(X_df, y_df, num_folds)

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


def var_plot(pca, scree=True):
    """
    Takes a data frame and pca value and generates a plot
    to show variance accounted for by principle components.

    :param pca: PCA object
    :param scree: True for scree plot, False for simple bar graph

    :return: none
    """

    var_ratio = pca.explained_variance_ratio_
    num_components = np.arange(len(var_ratio))
    cum_vals = np.cumsum(var_ratio)

    plt.figure(figsize=(15, 6))
    ax = plt.subplot(111)

    ax.bar(num_components, var_ratio)
    if scree:
        ax.plot(num_components, cum_vals, color='r')

    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    plt.title('Explained Variance Per Principal Component')
