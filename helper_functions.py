import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression


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
    print(f'R-squared:     {r2:.4}')
    print(f'RMSE:          {rmse:.4}')


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
    avg_r2 = np.mean(
            -1 * cross_val_score(
            model,
            X_df,
            y_df,
            scoring='r2',
            cv=KFold(n_splits=num_folds)
        )
    )

    avg_rmse = np.mean(
            -1 * cross_val_score(
            model,
            X_df,
            y_df,
            scoring='neg_root_mean_squared_error',
            cv=KFold(n_splits=num_folds)
        )
    )

    return avg_r2, avg_rmse


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


def component_reduction(X_df, y_df, model, max_components=50, pca=True):
    """
    Use PCA or PLS to return list with average RMSE for all number of components tested

    :param X_df: dataframe, usually X_train
    :param y_df: dataframe, usually y_train
    :param model: object, Regression model chosen
    :param max_components: int, Maximum number of components
    :param pca: bool, True: Fit PCA; False: Fit PLS

    :return:
    """
    # Fit for PCA
    if pca:
        comp_reduction = PCA(random_state=42)

        X_reduced = comp_reduction.fit_transform(X_df)

        avg_rmses = []

        # Check the first 25 principle components for RMSE
        for i in np.arange(1, max_components+1):
            # Resets model
            curr_model = model
            _, rmse = kfold_validate_score(X_reduced[:, :i], y_df, model=curr_model)
            avg_rmses.append(rmse)

    # Fit for PLS
    else:
        avg_rmses = []

        for i in np.arange(1, max_components+1):
            pls = PLSRegression(n_components=i)

            # Fit and transform
            X_train_pls = pls.fit(X_df, y_df)
            X_train_pls = pls.transform(X_df)

            curr_model = model

            _, rmse = kfold_validate_score(X_train_pls, y_df, model=curr_model)

            avg_rmses.append(rmse)

    return avg_rmses


def plot_components(primary_rmses, secondary_rmses=None, baselines=None):
    """

    :param primary_rmses:
    :param secondary_rmses:
    :param baselines:
    :return:
    """
    colors = ['r', 'grey', 'pink', 'purple']
    # var to iterate through color
    color = 0

    # plot primary rmses
    plt.plot(primary_rmses, color='blue')

    # plot secondary rmses if any
    if secondary_rmses:
        plt.plot(primary_rmses, color='orange')
        plt.title(f'{primary_rmses} vs. {secondary_rmses}')
    else:
        plt.title(f'{primary_rmses}')

    # add baselines if any
    if baselines:
        for key, val in baselines.items():
            plt.axhline(val, color=colors[color], label=f'{key} baseline')
            # increment color var
            color += 1

        plt.legend()

    plt.xlabel('Number of Components')
    plt.ylabel('Average RMSE');


def add_baselines(baselines):
    """

    :param baselines:
    :return:
    """
    colors = ['r', 'grey', 'pink', 'purple']
    # var to iterate through color
    color = 0

    for key, val in baselines.items():
        plt.axhline(val, color=colors[color], label=f'{key} baseline')
        # increment color var
        color += 1

    plt.legend()
