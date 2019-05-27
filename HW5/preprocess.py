"""
HW5: preprocess
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

    :return: a data frame with outcome
    '''

    df[post_col] = pd.to_datetime(df[post_col])
    df[funded_col] = pd.to_datetime(df[funded_col])
    df[interval_col] = df[funded_col] - df[post_col]
    df[outcome_col] = np.where(df[interval_col] <= period, 0, 1)
    return df.drop(columns=[interval_col])


def imputation(X_train, X_test, colnames, is_num=True):
    '''
    imputate the cells that are NaN with intended value. If it is numeric, then
    imputating value is the mean of training data, else it would be 'unknown'.
    :param X_train: a data frame of training set
    :param X_test: a data frame of test set
    :param colnames:(list) a list of colnames to be imputated
    :param is_num: (bool) check whether those columns are numeric or not

    :return: No returns.
    '''
    for colname in colnames:
        if is_num:
            impute_val = X_train[colname].mean()
        else:
            impute_val = 'unknown'

        X_train[colname] = X_train[colname].fillna(value=impute_val)
        X_test[colname] = X_test[colname].fillna(value=impute_val)



def discritize(X_train, X_test, colname, labels_list):
    '''
    Discritize the continuous variable  based on training data
    Inputs:
        X_train: a data frame of training set
        X_test: a data frame of test set
        colname: the name of the column
        labels_list: the label of
    Output:
        add a new column to train ans test set respectively
        that are discritized from a continuous variable
    '''
    n = len(labels_list)
    quantile_list = []
    for i in range(0, n + 1):
        quantile_list.append(i / n)
    bins_list = list(X_train[colname].quantile(quantile_list).values)
    bins_list[0] = bins_list[0] - 1
    
    X_train[(colname + '_category')] = pd.cut(X_train[colname],
                                       bins=bins_list,
                                       labels=labels_list)
    X_test[(colname + '_category')] = pd.cut(X_test[colname],
                                      bins=bins_list,
                                      labels=labels_list)


def get_all_dummies(X_train, X_test, colname):
    '''
    Convert the categorical variable into dummies
    Inputs:
        X_train: a data frame of training set
        X_test: a data frame of test set
        colname: the name of the colname
    Return:
        the data frame with those dummies into data frame
    '''
    #Get the categories from training data set
    cat_list = list(X_train[colname].value_counts().index.values)
    #create dummies
    for cat in cat_list:
        X_test[cat] = np.where(X_test[colname] == cat, 1, 0)
        X_train[cat] = np.where(X_train[colname] == cat, 1, 0)

    
def get_top_k_dummies(X_train, X_test, colname, k):
    '''
    For columns with too many categories, only create dummies for 
    top k categories
    Inputs:
        X_train: a data frame of training set
        X_test: a data frame of test set
        colname: the name of the column
        k: (int) the value of k
    Outputs:
       Create dummies in both train and test set
    '''
    # get top k categories from tarin set
    top_k = X_train[colname].value_counts()[:k].index
    #create dummies
    for cat in top_k:
        X_train[cat] = np.where(X_train[colname] == cat, 1, 0)
        X_test[cat] = np.where(X_test[colname] == cat, 1, 0)
    X_train['{}_others'.format(colname)] = X_train.apply(\
        lambda x: 0 ifx[colname] in top_k else 1, axis=1)
    X_test['{}_others'.format(colname)] = X_test.apply(\
        lambda x: 0 if x[colname] in top_k else 1, axis=1)
        

def get_dummies(X_train, X_test, colname, k):
    '''
    Wrap up get_all_dummies and get_top_k_dummies
    Inputs:
        X_train: a data frame of training set
        X_test: a data frame of test set
        colname: the name of the column
        k: (int) the value of k
    Outputs:
       Create dummies in both train and test set
    '''
    #Decide whether this use get all dummies or top k
    if len(X_train[colname].value_counts()) > k:
        get_top_k_dummies(X_train, X_test, colname, k)
    else:
        get_all_dummies(X_train, X_test, colname)
        