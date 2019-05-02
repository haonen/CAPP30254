"""
HW3: preprocess
Yuwei Zhang
"""

import pandas as pd
import numpy as np


def create_outcome(df, post_col, funded_col, interval_col, outcome_col, period):
    '''
    Define the outcome column
    :param df: a data frame
    :param post_col: (str) the name of the posted date
    :param funded_col: (str) the name of the funded date
    :param interval_col: (str) the name of the interval
    :param outcome_col: (str) the name of the outcome column
    :param period: (timedelta object) the period that the project receive funding

    :return: No returns but add three two columns
    '''

    df[post_col] = pd.to_datetime(df[post_col])
    df[funded_col] = pd.to_datetime(df[funded_col])
    df[interval_col] = df[funded_col] - df[post_col]
    df[outcome_col] = np.where(df[interval_col] <= period, 1, 0)


def imputation(df, colnames, is_num=True):
    '''
    imputate the cells that are NaN with intended value. If it is numeric, then
    imputating value is its mean, else it would be 'unknown'.
    :param df: a data frame
    :param colnames:(list) a list of colnames to be imputated
    :param is_num: (bool) check whether those columns are numeric or not

    :return: No returns.
    '''

    for colname in colnames:
        if is_num:
            impute_val = df[colname].mean()
        else:
            impute_val = 'unknown'

        df[colname] = df[colname].fillna(value=impute_val)


def discritize(df, colname, bins_list, labels_list):
    '''
    Discritize the continuous variable
    Inputs:
        df: a data frame
        colname: the name of the column
        bins_list: the list of the boundaries to be cut
        labels_list: the label of
    Output:
        add a new column that are discritized from a continuous variable
    '''
    df[(colname + '_category')] = pd.cut(df[colname],
                         bins=bins_list,
                         labels=labels_list,
                         include_lowest=True, right=False)


def get_dummies(df, colname):
    '''
    Convert the categorical variable into dummies
    Inputs:
        df: a data frame
        colname: the name of the colname
    Return:
        the data frame with those dummies into data frame
    '''
    return pd.concat([df,pd.get_dummies(df[colname])], axis=1)
