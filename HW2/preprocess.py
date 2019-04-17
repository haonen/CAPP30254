"""
HW2: Data Preprocessing
Yuwei Zhang
"""

import pandas as pd


def imputation(df, colname):
    '''
    imputate the cells that are NaN with the mean of this column
    Inputs:
        df: a data frame
        colname: the name of the colunmn that contains NA values
    Output:
        imputate NA cells with mean value
    '''
    avg = df[colname].mean()
    df[colname] = df[colname].fillna(value=avg)


def discretize(df, colname, bins_list, labels_list):
    '''
    Discretize the continuous variable
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
