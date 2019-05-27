"""
HW5: modeling
Yuwei Zhang
"""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import ParameterGrid
from datetime import timedelta
from datetime import datetime
from evaluation import *


def split_data(X, y, test_size, temporal, split_var, start_date, end_date, delay_interval, period):
    '''
    Split the data set
    :param X: the feature set
    :param y: the outcome set
    :param test_size: (float) the size of test
    :param temporal: (boolean) whether to use temporal validation
    :param split_var: (str) split the data set based on this variable
    :param start_date: the start date of this data set
    :param end_date: the end date of this data set
    :param delay_interval: how many days the test set should be away from train set
    :param period: (relativedelta object) the period

    :return: a list of splited train and test sets
    '''
    one_day = timedelta(days=1)
    split_results = []
    if temporal:
        count = 1
        while True:
            train_end = start_date + period * count - one_day - delay_interval
            X_train = X[X[split_var] < train_end]
            X_train = X_train.drop([split_var], axis=1)
            y_train = y[X[split_var] < train_end]
            test_start = start_date + period * count
            test_end = test_start + period - one_day
            X_test = X[(X[split_var] >= test_start) & (X[split_var] <= test_end)]
            X_test = X_test.drop([split_var], axis=1)
            y_test = y[(X[split_var] >= test_start) & (X[split_var] <= test_end)]
            split_results.append((X_train, X_test, y_train, y_test))
            count += 1
            print('start date: {}'.format(start_date))
            print('train_end_date: {}'.format(train_end))
            print('test_start_date: {}'.format(test_start))
            print('test_end_date: {}'.format(test_end))
            print('end_date: {}'.format(end_date))
            print()
            # when exhaust the last date of project, stop the loop
            if start_date + period * count - one_day >= end_date:
                break
    else:
        X = X.drop([temporal_var], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= \
            test_size, random_state=0)
        split_results.append((X_train, X_test, y_train, y_test))
    return split_results


def run_models(models_list, clfs, grid, X_train, X_test, y_train, y_test, threshold):
    """
    Run all the models in models_list and adjust hyper-parameters based on grid, evaluate
    them, save the results into a data frame and save the corresponding graphs
    Inputs:
        models_list:(list of str)a list of names of models to run
        clfs:a dictionary of base models
        grid:a dictionary of parameters
        X_train:the train feature set
        X_test: the test feature set
        y_train: the train outcome set
        y_test: the test outcome set
        threshold: (int) threshold for evaluation metircs
    Returns:
        a data frame and a bunch of graphs
    """
    # create the empty data frame
    col_list = ['model_name', 'parameters', 'baseline', 'accuarcy', 'f1', 'auc_roc',
                'precision_1%', 'precision_2%', 'precision_5%',
                'precision_10%', 'precision_20%', 'precision_30%',
                'precision_50%', 'recall_1%', 'recall_2%',
                'recall_5%', 'recall_10%', 'recall_20%','recall_30%', 'recall_50%' ]
    results_df =  pd.DataFrame(columns=col_list)
    
    for index,clf in enumerate([clfs[x] for x in models_list]):
        parameter_values = grid[models_list[index]]
        for params in ParameterGrid(parameter_values):
            try:
                print("Running {} ...".format(models_list[index]))
                clf.set_params(**params)
                if models_list[index] == 'Support Vector Machine':
                    y_pred_probs = clf.fit(X_train, y_train).decision_function(X_test)
                else:
                    y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
                
                if models_list[index] == "Decision Tree" or models_list[index] == "Random Forest":
                    d = {'Features': X_train.columns, "Importance": clf.feature_importances_}
                    feature_importance = pd.DataFrame(data=d)
                    feature_importance = feature_importance.sort_values(by=['Importance'], ascending=False)
                    print(feature_importance.head())
                
                # Sort true y labels and predicted scores at the same time    
                y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
                # Write the evaluation results into data frame
                results_df.loc[len(results_df)] = [models_list[index], params,
                                                   precision_at_k(y_test_sorted,y_pred_probs_sorted, 100),
                                                   compute_acc(y_test_sorted, y_pred_probs_sorted, threshold),                                                              compute_f1(y_test_sorted, y_pred_probs_sorted, threshold),
                                                   compute_auc_roc(y_test_sorted, y_pred_probs_sorted, threshold),
                                                   precision_at_k(y_test_sorted,y_pred_probs_sorted,1),
                                                   precision_at_k(y_test_sorted,y_pred_probs_sorted,2),
                                                   precision_at_k(y_test_sorted,y_pred_probs_sorted,5),
                                                   precision_at_k(y_test_sorted,y_pred_probs_sorted,10),
                                                   precision_at_k(y_test_sorted,y_pred_probs_sorted,20),
                                                   precision_at_k(y_test_sorted,y_pred_probs_sorted,30),
                                                   precision_at_k(y_test_sorted,y_pred_probs_sorted,50),
                                                   recall_at_k(y_test_sorted,y_pred_probs_sorted,1),
                                                   recall_at_k(y_test_sorted,y_pred_probs_sorted,2),
                                                   recall_at_k(y_test_sorted,y_pred_probs_sorted,5),
                                                   recall_at_k(y_test_sorted,y_pred_probs_sorted,10),
                                                   recall_at_k(y_test_sorted,y_pred_probs_sorted,20),
                                                   recall_at_k(y_test_sorted,y_pred_probs_sorted,30),
                                                   recall_at_k(y_test_sorted,y_pred_probs_sorted,50)]
                
                graph_name_pr = 'D:/UChicago/2019 spring/CAPP30254/assignments/HW5/grpahs/' + \
                                'precision_recall_curve of ' + models_list[index] + \
                                datetime.now().strftime("%m-%d-%Y %H%M%S")
                plot_precision_recall_n(y_test, y_pred_probs, clf, graph_name_pr, 'save')
                graph_name_roc = 'D:/UChicago/2019 spring/CAPP30254/assignments/HW5/grpahs/' + \
                                 'roc_curve of' + models_list[index] +\
                                 datetime.now().strftime("%m-%d-%Y  %H%M%S")
                plot_roc(clf, graph_name_roc, y_pred_probs, y_test, 'save')
            except IndexError as e:
                print('Error:',e)
                continue
    return results_df
