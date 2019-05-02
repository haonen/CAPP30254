"""
HW3: modeling
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


def split_data(X, y, test_size, temporal, split_var, split_period_list, period):
    '''
    Split the data set
    :param X: the feature set
    :param y: the outcome set
    :param test_size: (float) the size of test
    :param temporal: (boolean) whether to use temporal validation
    :param split_var: (str) split the data set based on this variable
    :param split_period_list: (list) a list of datetime object
    :param period: (relativedelta object) the period

    :return:
    '''

    split_results = []
    if temporal:
       for i, date in enumerate(split_period_list):
           next_date = date + period
           X_train = X[X[split_var] <= date]
           X_train = X_train.drop([split_var], axis=1)
           y_train = y[X[split_var] <= date]
           X_test = X[(X[split_var] > date) & (X[split_var] <= next_date)]
           X_test = X_test.drop([split_var], axis=1)
           y_test = y[(X[split_var] > date) & (X[split_var] <= next_date)]
           split_results.append((X_train, X_test, y_train, y_test))
    else:
        X = X.drop([temporal_var], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= \
            test_size, random_state=0)
        split_results.append((X_train, X_test, y_train, y_test))
    return split_results


def build_lr(param_dict, X_train, y_train):
    '''
    Build a Logistic Regression model based on different choices of parameters
    :param X_train: feature set for training
    :param y_train: outcome set for training
    :param param_dict: (dict) a dictionary about parameters

    :return: a dictionary of logistic regression models
    '''
    model_dict = {}
    solver_list = param_dict['solver']
    penalty_list = param_dict['penalty']
    C_list = param_dict['C']
    for solver_choice in solver_list:
        for penalty_choice in penalty_list:
            for c_val in C_list:
                key = "solver={}, penalty={}, C={}".format(solver_choice, penalty_choice, c_val)
                lr = LogisticRegression(solver=solver_choice, penalty=penalty_choice, C=c_val)
                lr.fit(X_train, y_train)
                model_dict[key] = lr

    return model_dict


def build_knn(param_dict, X_train, y_train):
    '''
    Build a k nearest neighbors model based on different choices of parameters
    :param param_dict: (dict) a dictionary about parameters
    :param X_train: feature set for training
    :param y_train: outcome set for training

    :return: a dictionary of knn models
    '''
    model_dict = {}
    neighbor_list = param_dict['n_neighbors']
    p_list = param_dict['p']
    for num_of_neighbors in neighbor_list:
        for p_val in p_list:
            key = "n_neighbors={}, p={}".format(num_of_neighbors, p_val)
            knn = KNeighborsClassifier(n_neighbors=num_of_neighbors, p=p_val)
            knn.fit(X_train, y_train)
            model_dict[key] = knn
    return model_dict


def build_dt(param_dict, X_train, y_train):
    '''
    Build a decision tree model based on different choices of parameters
    :param param_dict: (dict) a dictionary about parameters
    :param X_train: the feature set for training
    :param y_train: the outcome set for training

    :return: a dictionary of decision tree models
    '''
    model_dict = {}
    criterion_list = param_dict['criterion']
    d_list = param_dict['max_depth']
    for criterion_choice in criterion_list:
        for d in d_list:
            key = "criterion={}, max_depth={}".format(criterion_choice, d)
            decision_tree = DecisionTreeClassifier(criterion=criterion_choice, max_depth=d)
            decision_tree.fit(X_train, y_train)
            model_dict[key] = decision_tree
    return model_dict


def build_svm(param_dict, X_train, y_train):
    '''
    Build a svm model based on different choices of parameters
    :param param_dict: (dict) a dictionary about parameters
    :param X_train: the feature set for training
    :param y_train: the outcome set for training

    :return: a dictionary of svm models
    '''
    model_dict = {}
    c_list = param_dict['C']
    for c_val in c_list:
        key = "C={}".format(c_val)
        svm = LinearSVC(random_state=0, tol=1e-5, C=c_val)
        svm.fit(X_train, y_train)
        model_dict[key] = svm
    return model_dict


def build_rf(param_dict, X_train, y_train):
    '''
    Build random forest models ensemble based on different choice
    of parameters
    :param param_dict: (dict) a dictionary about parameters
    :param X_train: the feature set for training
    :param y_train: the outcome set for training

    :return: a dictionary of random forest models
    '''
    model_dict = {}
    estimator_list = param_dict['n_estimators']
    d_list = param_dict['max_depth']
    criterion_list = param_dict['criterion']
    for num_estimators in estimator_list:
        for d in d_list:
            for criterion_choice in criterion_list:
                key = "n_estimators={}, max_depth={}, criterion={}".format(num_estimators, d, criterion_choice)
                rf = RandomForestClassifier(n_estimators=num_estimators,
                                            max_depth=d,
                                            criterion=criterion_choice)
                rf.fit(X_train, y_train)
                model_dict[key] = rf
    return model_dict


def build_bagging(param_dict, X_train, y_train):
    '''
    Build bagging models from different parameters
    :param param_dict: (dict) a dictionary about parameters
    :param X_train: the feature set for training
    :param y_train: the outcome set for training

    :return: a dictionary of bagging models
    '''
    model_dict = {}
    base_list = param_dict['base_estimator']
    estimator_list = param_dict['n_estimators']
    sample_list = param_dict['max_samples']
    for base in base_list:
        for num_estimators in estimator_list:
            for sample in sample_list:
                key = "base_estimator={}, n_estimators={}, max_samples={}".format(base, num_estimators, sample)
                bagging = BaggingClassifier(base_estimator=base,
                                            n_estimators= num_estimators,
                                            max_samples=sample)
                bagging.fit(X_train, y_train)
                model_dict[key] = bagging
    return model_dict


def build_boosting(param_dict,X_train, y_train):
    '''
    Build a boosting model based on different choices of parameters
    :param param_dict: (dict) a dictionary about parameters
    :param X_train: (np array) the feature set for training
    :param y_train: (np.array) the outcome set for training

    :return: a list of boosting models
    '''
    model_dict = {}
    base_list = param_dict['base_estimator']
    estimator_list = param_dict['n_estimators']
    rate_list = param_dict['learning_rate']
    for base in base_list:
        for num_estimators in estimator_list:
            for rate in rate_list:
                key = "base_estimator={}, n_estimators={}, learning_rate={}".format(base, num_estimators, rate)
                boosting = AdaBoostClassifier(base_estimator=base,
                                              n_estimators=num_estimators,
                                              learning_rate=rate)
                boosting.fit(X_train, y_train)
                model_dict[key] = boosting
    return model_dict


def predict_models(classifier, model_dict, X_test):
    pred_scores_dict = {}
    for params, model in model_dict.items():
        if classifier == 'Support Vector Machine':
            pred_scores = model.decision_function(X_test)
        else:
            pred_scores = model.predict_proba(X_test)[:, 1]
        pred_scores_dict[params] = pred_scores
    return pred_scores_dict



