"""
HW 2: Modeling and evaluation
Yuwei Zhang
"""

import numpy as np
from sklearn import tree
from sklearn import preprocessing
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def preprocess_feature(df, features):
    '''
    Choose features with selected columns and normalize them
    Inputs:
        df: a data frame
        features: a list of selected feature names
    Return:
        preprocessed features
    '''
    X = df[features]
    X = preprocessing.StandardScaler().fit(X).transform(X)
    return X


def split_data(X, y, test_size):
    '''
    Split data set into train set and test set. Print out their shape
    Inputs:
        X: feature array
        y: an numpy array
        test_size: the proportion of test set
    Returns:
        train set and test set
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=4)
    print('Train set:', X_train.shape,  y_train.shape)
    print('Test set:', X_test.shape,  y_test.shape)
    return X_train, X_test, y_train, y_test


def get_label(pred_score, threshold):
    '''
    Get the label of test data based on choosen threshold
    Inputs:
        pred_score(numpy array): an arrary of predicted probability
        threshold(float): the threshold to choose labels
    Returns:
        an array of predicted labels
    '''
    pred_label = np.array(list(map(lambda x: 1 if x > threshold else 0, pred_score)))
    return pred_label


def fit_and_evaluation(x_train, y_train, x_test, y_test):
    '''
    Create the decision tree model with max_depth from 3 to 6 and evaluate each model
    by jaccard index and F1 score report
    Inputs:
       train set and test set
    Output:
        print out differnt evaluation results for differnt models
        return the best decision tree model
    '''
    max_jaccard = float('-inf')
    best_tree = None
    for d in range(3, 7):
        creditTree = tree.DecisionTreeClassifier(criterion="entropy", max_depth=d)
        creditTree.fit(x_train, y_train)
        # Predicting
        DT_score = creditTree.predict_proba(x_test)[:,1]
        DT_yhat = get_label(DT_score, 0.4)
        print('evaluation for max_depth = {}:'.format(d))
        print()
        # Jaccard Index
        j_index = jaccard_similarity_score(y_test, DT_yhat)
        print("    jaccard index for DT: {}".format(round(j_index, 3)))
        # F1_score
        print("    F1 score report for DT:" + "\n", classification_report(y_test, DT_yhat))
        
        if j_index >= max_jaccard:
            max_jaccard = j_index
            best_tree = creditTree
    return best_tree
