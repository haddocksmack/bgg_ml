import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression


def rename_columns(df, df_name):
    """
    Adds the dataframe name to column names.
    This helps with analyzing features later

    :param df: dataframe object, dataframe to rename columns
    :param df_name: str, name of dataframe

    :return: dataframe object, dataframe with renamed columns
    """
    old_names = df.columns.to_list()

    new_names = {}

    for i in range(len(old_names)):
        # exclude `game_id`
        if i != 0:
            new_names[old_names[i]] = str(old_names[i]) + f' ({df_name})'
        else:
            new_names[old_names[i]] = old_names[i]

    df.rename(columns=new_names, inplace=True)

    return df


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
        comp_reduction = PCA(random_state=37)

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


def add_baselines(baselines):
    """
    Plots baseline values

    :param baselines: dictionary, dict of {baseline: RMSE score}

    :return: None
    """
    colors = ['r', 'grey', 'pink', 'purple']
    # var to iterate through color
    color = 0

    for key, val in baselines.items():
        plt.axhline(val, color=colors[color], label=f'{key} baseline')
        # increment color var
        color += 1

    plt.legend()


def pls_comp_list(df, comp_num, threshold=0.10):
    """
    Returns a df of the most important features of a game by PLS component X_weights

    :param df: dataframe object, dataframe of X_weights generated by PLS Regression
    :param comp_num: int, chosen PLS component
    :param threshold: float, threshold of X_weight to include in output df

    :return: dataframe object, dataframe of the most important features by weight of a chosen PLS Component
    """
    # convert comp_num to string
    comp_num = str(comp_num)

    # get needed col names
    comp_col = 'comp_' + comp_num
    abs_col = 'abs_' + comp_col

    # sort df by chosen component abs col
    df = df.sort_values(by=abs_col, ascending=False)

    # drop all rows with vals lower than the threshold
    df = df[df[abs_col] >= threshold]

    return df[comp_col]
