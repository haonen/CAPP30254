"""
HW3: Analysis
Yuwei Zhang
"""

import pandas as pd
import modeling
import evaluation


def analysis(classifier_dict, model_name, X_train, y_train, X_test, y_test):
    '''
    Plot the precision-recall curve for best models
    :param classifier_dict: (dict) a dictionary for the best model
    :param model_name: (str) the name of model
    :param X_train: the train feature set
    :param y_train: the train outcome set
    :param X_test: the test feature set
    :param y_test: the test outcome set

    :return: save or show the precision recall curve
    '''
    for classifier, param_dict in classifier_dict.items():
        if classifier == 'Logistic Regression':
            model_dict = modeling.build_lr(param_dict, X_train, y_train)
        if classifier == 'K Nearest Neighbors':
            model_dict = modeling.build_knn(param_dict, X_train, y_train)
        if classifier == 'Decision Tree':
            model_dict = modeling.build_dt(param_dict, X_train, y_train)
        if classifier == 'Support Vector Machine':
            model_dict = modeling.build_svm(param_dict, X_train, y_train)
        if classifier == 'Random Forest':
            model_dict = modeling.build_rf(param_dict, X_train, y_train)
        if classifier == 'Boosting':
            model_dict = modeling.build_boosting(param_dict, X_train, y_train)
        if classifier == 'Bagging':
            model_dict = modeling.build_bagging(param_dict, X_train, y_train)

        pred_scores_dict = modeling.predict_models(classifier, model_dict, X_test)
        for params, pred_score in pred_scores_dict.items():
            evaluation.plot_precision_recall_n(y_test, pred_score, model_name, 'save')


def get_feature_importance(classifier_dict, X_train, y_train):
    '''
    Create the data frame for feature's importance of decision tree models
    :param classifier_dict: (dict) a dictionary for the best model
    :param X_train: the train feature set
    :param y_train: the train outcome set

    :return: a data frame about features' importance
    '''
    for classifier, param_dict in classifier_dict.items():
        model_dict = modeling.build_rf(param_dict, X_train, y_train)
        for _, model in model_dict.items():
            d = {'Features': X_train.columns, "Importance": model.feature_importances_}
            feature_importance = pd.DataFrame(data=d)
            feature_importance = feature_importance.sort_values(by=['Importance'], ascending=False)
        return feature_importance
