import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def full_training_set_scores(X_df, y_df, model):
    """
    Predicts on the same data set a model is fit to
    Returns R-squared and RMSE scores

    :param X_df: X dataframe, usually X_train
    :param y_df: y dataframe, usually y_train
    :param model: model used for fitting and predicting

    :return: R-squared and RMSE scores
    """
    model = model
    preds = model.predict(X_df)

    r2 = r2_score(y_df, preds)
    rmse = mean_squared_error(y_df, preds, squared=False)

    return r2, rmse


def print_scores(r2, rmse, model_name=None):
    """
    Prints scores from a model

    :param r2: float, R-squared score
    :param rmse: float, RMSE score
    :param model_name: string, name of model used for scoring

    :return: None
    """
    if model_name:
        print(f'{model_name}:\n')
    print(f'R-squared:     {r2}')
    print(f'RMSE:          {rmse}')


def kfold_validate_pred(X_df, y_df, model, num_folds=8):
    """
    Perform k-fold cross validation and return predictions

    :param X_df: X dataframe, usually X_train
    :param y_df: y dataframe, usually y_train
    :param num_folds: number of folds used in KFolds()

    :return: numpy array containing predictions from Linear Regression
             K-Fold Cross Validation
    """
    cv = KFold(n_splits=num_folds)

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
    training = -1 * cross_val_score(
        model,
        X_df,
        y_df,
        scoring='neg_root_mean_squared_error',
        cv=KFold(n_splits=num_folds)
    ).mean()

    avg_rmse = mean_squared_error(y_df, model.predict(X_df), squared=False)

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
