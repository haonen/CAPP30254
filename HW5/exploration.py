"""
HW5: data exploration
Yuwei Zhang
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def read_data(path, index_col):
    '''
    read the csv file
    Inputs:
        path: the path of the csv file
        index_col: the name of the index column
    Return:
        the data frame
    '''
    df = pd.read_csv(path, index_col=index_col)
    return df


def plot_count(df, colname, hue_col):
    '''
    Plot the counting plot of categorical variable
    Inputs:
        df: a data frame
        colname: the name of a column
    Output:
        plot the counting for different categories
    '''
    sns.set(style="darkgrid")
    sns.countplot(x=colname, hue=hue_col, data=df).set_title(('Distribution of ' + colname))
    plt.show()


def plot_pair(df):
    '''
    Plot the pairwise relationship in a dataset
    Inputs:
        df: a data frame
    Ouput:
        plot the pair plot
    '''
    sns.set(font_scale=1)
    sns.pairplot(df, height=3)
    plt.tight_layout()


def plot_heatmap(df):
    '''
    Plot the correlation matrix of features and outcome in a heat map
    Inputs:
        df: a data frame
    Output:
        a heatmap
    '''
    cm = df.corr(method='pearson')
    sns.set(font_scale=1.5)
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, cbar=True,
                annot=False,
                square=True,
                fmt='.2f',
                annot_kws={'size': 10},
                yticklabels=df.columns,
                xticklabels=df.columns, ax=ax)

    
def plot_dist(df, colname):
    '''
    Plot the distribution plot of a variable
    Inputs:
        df: a data frame
        colname: the name of the colunm
    Output:
        a distribution plot
    '''
    sns.distplot(df[colname])
    plt.show()