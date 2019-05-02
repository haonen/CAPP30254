"""
HW3: evaluation
Yuwei Zhang
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from modeling import *


def compute_acc(y_true, y_scores, k):
    '''
    Compute accuracy score based on threshold
    :param pred_scores: (np array) an array of predicted score
    :param threshold: (float) the threshold of labeling predicted results
    :param y_test: test set

    :return: (float) an accuracy score
    '''
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)

    return accuracy_score(y_true_sorted, preds_at_k)


def compute_f1(y_true, y_scores, k):
    '''
    Compute f1 score based on threshold
    :param pred_scores: (np array) an array of predicted score
    :param threshold: (float) the threshold of labeling predicted results
    :param y_test: test set

    :return: (float) an f1 score
    '''
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)

    return f1_score(y_true_sorted, preds_at_k)

def compute_auc_roc(y_true, y_scores, k):
    '''
    Compute area under Receiver Operator Characteristic Curve
    :param pred_scores: (np array) an array of predicted score
    :param threshold: (float) the threshold of labeling predicted results
    :param y_test: test set

    :return: (float) an auc_roc score
    '''
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)

    return roc_auc_score(y_true_sorted, preds_at_k)


def compute_auc(pred_scores, true_labels):
    '''
    Compute auc score
    :param pred_scores: an array of predicted scores
    :param true_labels: an array of true labels

    :return: area under curve score
    '''
    fpr, tpr, thresholds = roc_curve(true_labels, pred_scores, pos_label=2)
    return auc(fpr, tpr)


def wrap_up(classifier_dict, X_train, y_train, X_test, y_test, threshold, threshold_list):
    '''
    A wrap up function to train data on al the intended classifiers and
    calculate the corresponding metrics
    :param classifier_dict: (dict) a dictionary of mapping classifiers to their parameters
    :param X_train: the train feature set
    :param y_train: the train outcome set
    :param X_test: the test feature set
    :param y_test: the test outcome set
    :param threshold: (float) the threshold for computing accuracy, f1, auc_roc
    :param threshold_list: (list) a list of thresholds for computing precision and recall

    :return: a dictionary of mapping models to evaluation metrics.
    '''
    count = 0
    evaluation = {}
    baseline = y_test.mean()
    for classifier, param_dict in classifier_dict.items():
        if classifier == 'Logistic Regression':
            model_dict = build_lr(param_dict, X_train, y_train)
        if classifier == 'K Nearest Neighbors':
            model_dict = build_knn(param_dict, X_train, y_train)
        if classifier == 'Decision Tree':
            model_dict = build_dt(param_dict, X_train, y_train)
        if classifier == 'Support Vector Machine':
            model_dict = build_svm(param_dict, X_train, y_train)
        if classifier == 'Random Forest':
            model_dict = build_rf(param_dict, X_train, y_train)
        if classifier == 'Boosting':
            model_dict = build_boosting(param_dict, X_train, y_train)
        if classifier == 'Bagging':
            model_dict = build_bagging(param_dict, X_train, y_train)

        pred_scores_dict = predict_models(classifier, model_dict, X_test)

        for params, score in pred_scores_dict.items():
            evaluation[count] = [classifier]
            print("Running {} ...".format(classifier))
            evaluation[count].append(params)
            evaluation[count].append(baseline)
            accuracy = compute_acc(y_test, score, threshold)
            evaluation[count].append(accuracy)
            f1 = compute_f1(y_test, score, threshold)
            evaluation[count].append(f1)
            auc_roc = compute_auc_roc(y_test, score, threshold)
            evaluation[count].append(auc_roc)
            for threshold in threshold_list:
                precision = precision_at_k(y_test, score, threshold)
                evaluation[count].append(precision)
                recall = recall_at_k(y_test, score, threshold)
                evaluation[count].append(recall)
            count += 1
    return evaluation


def write_as_df(dict, col_list):
    '''
    Write the evaluation result into a data frame
    :param dict: a dictionary mapping models to evaluation results
    :param col_list: (list) a list of the column names in data frame

    :return: a data frame
    '''
    return pd.DataFrame.from_dict(dict, orient='index', columns=col_list)


# The following functions are referenced from:
# https://github.com/rayidghani/magicloops/blob/master/mlfunctions.py

def joint_sort_descending(l1, l2):
    '''
    Sort two arrays together
    :param l1:  numpy array
    :param l2:  numpy array

    :return: two sorted arrays
    '''
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]


def generate_binary_at_k(y_scores, k):
    '''
    predict labels based on thresholds
    :param y_scores: the predicted scores
    :param k: (int or float) threshold

    :return: predicted labels
    '''
    cutoff_index = int(len(y_scores) * (k / 100.0))
    predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return predictions_binary


def precision_at_k(y_true, y_scores, k):
    '''
    Compute precision based on threshold (percentage)
    :param y_true: the true labels
    :param y_scores: the predicted labels
    :param k: (int or float) the threshold

    :return: (float) precision score
    '''
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    return precision_score(y_true_sorted, preds_at_k)


def recall_at_k(y_true, y_scores, k):
    '''
    Compute recall based on threshold (percentage)
    :param y_true: the true labels
    :param y_scores: the predicted labels
    :param k: (int or float) the threshold

    :return: (float) recall score
    '''
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    return recall_score(y_true_sorted, preds_at_k)


def plot_precision_recall_n(y_true, y_prob, model_name, output_type):
    '''
    Plot precision and recall at different percent of population
    :param y_true: the true labels
    :param y_prob: the predicted labels
    :param model_name: the name of the model
    :param output_type: (str) 'save' or 'show'

    :return: No returns but a plot
    '''
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score >= value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)

    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0, 1])
    ax1.set_ylim([0, 1])
    ax2.set_xlim([0, 1])

    name = model_name
    plt.title(name)
    if (output_type == 'save'):
        plt.savefig(name)
    elif (output_type == 'show'):
        plt.show()
    else:
        plt.show()
