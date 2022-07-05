"""
This file is to call algorithms for solving the problems of classification
authored: Phuong V. Nguyen
"""
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
# pre-process
from sklearn.model_selection import train_test_split
# Linear algorithm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
# Non-linear algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
# Gradient Boosting Decision Tree family
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
# Neural Networks
from sklearn.neural_network import MLPClassifier
# Metric
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.pipeline import Pipeline
from typing import List, Tuple, Any, Union, Optional, Dict


warnings.filterwarnings("ignore")


def model_list():
    """
    this function is to call a list of potential models to compete
    :return models: list of models
    """
    models = [('Logistic', LogisticRegression(max_iter=1000)),
              ('RidgeClassifier', RidgeClassifier()),
              ('LDA', LinearDiscriminantAnalysis()),
              ('QDA', QuadraticDiscriminantAnalysis()),
              ('SGDC', SGDClassifier()),
              ('GNB', GaussianNB()),
              ('SVC', SVC()),
              ('KNN', KNeighborsClassifier()),
              ('DT', DecisionTreeClassifier()),
              ('ADBoost', AdaBoostClassifier()),
              ('GBBoost', GradientBoostingClassifier()),
              ('XGBoost', XGBClassifier(eval_metric='logloss')),
              ('RanForest', RandomForestClassifier()),
              ('ExTree', ExtraTreesClassifier())]
    return models


def plot_result(results, names):
    """
    This function is to plot the result of comparing performance among selected models
    :param results: score of performance
    :param names: name of competing model
    """
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle('Model Performance Comparision (AUROC)')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names, rotation=80)
    ax.grid(True)
    plt.show()


def models_comparer(list_of_model, x, y):
    """
    This function is to compare the performance of some selected models
    :param list_of_model: list of model for competition
    :param x: feature
    :param y: label
    :return results, names:
    """
    results = []
    names = []
    for name, model in list_of_model:
        kfold = KFold(n_splits=10, random_state=7)
        cv_results = cross_val_score(model, x, y.values.ravel(),
                                     cv=kfold, scoring='roc_auc')
        results.append(cv_results)
        names.append(name)
        msg = '%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())
        print(msg)
    plot_result(results, names)
    return results, names


def create_model(model_name: str == 'LogisticRegression',
                 x, y,
                 nsplit: int = 10,
                 ramstate: int = 7):
    kfold = KFold(n_splits=10, random_state=7)
    models = model_list()
    if model_name == 'LogisticRegression':
        created_model = models[0][1]
    elif model_name == 'RidgeClassifier':
        created_model = models[1][1]
    else:
        created_model = models[2][1]
    cv_results = cross_val_score(created_model, x, y,
                                 cv=kfold, scoring='roc_auc')
    meancv = cv_results.mean()
    sdcv = cv_results.std()
    result = pd.DataFrame({'AUC': cv_results})
    new_row_mean = pd.Series(data={'AUC': meancv}, name='Mean')
    new_row_std = pd.Series(data={'AUC': sdcv}, name='Std')
    result = result.append(new_row_mean, ignore_index=False)
    result = result.append(new_row_std, ignore_index=False)
    display(result)
    return created_model


def obj_logreg(trial, x, y):
    penalty = trial.suggest_categorical('penalty', ['l2', 'l1'])
    if penalty == 'l1':
        solver = 'saga'
    else:
        solver = 'lbfgs'
    regularization = trial.suggest_uniform('C', 0.01, 100)
    model = LogisticRegression(penalty=penalty,
                               C=regularization, solver=solver)
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, x, y,
                                 cv=kfold, scoring='roc_auc')
    aver_auc = cv_results.mean()
    return aver_auc


def obj_ridgec(trial,
               x: pd.DataFrame,
               y: pd.DataFrame):
    penalty = trial.suggest_uniform('alpha', 0.01, 100)
    optimizer = trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky',
                                                  'lsqr', 'sparse_cg', 'sag', 'saga'])
    model = RidgeClassifier(alpha=penalty, solver=optimizer)

    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, x, y,
                                 cv=kfold, scoring='roc_auc')
    aver_auc = cv_results.mean()
    return aver_auc


def tune_model(model_name: str == 'LogisticRegression',
               x_feat, y_target,
               num_trial: int = 100):
    study = optuna.create_study()
    if model_name == 'LogisticRegression':
        study.optimize(lambda tr: obj_logreg(tr, x=x_feat, y=y_target),
                       n_trials=num_trial)
        best_set = study.best_params
        if best_set['penalty']=='l1':
            tuned_model = LogisticRegression(C=best_set['C'],
                                             penalty=best_set['penalty'], solver='saga')
        else:
            tuned_model = LogisticRegression(C=best_set['C'],
                                             penalty=best_set['penalty'], solver='lbfgs')
    elif model_name == 'RidgeClassifier':
        study.optimize(lambda tr: obj_ridgec(tr, x=x_feat, y=y_target),
                       n_trials=num_trial)
        best_set = study.best_params
        tuned_model = RidgeClassifier(alpha=best_set['alpha'],
                                      solver=best_set['solver'])
    else:
        print('underconstrion')
    print('The configuration of tuned model:', tuned_model, sep='\n')
    tuned_model.fit(x_feat, y_target)
    return tuned_model


def discrim_power(x_test: pd.DataFrame,
               y_test: pd.DataFrame, trained_model):
    y_prob_pred=trained_model.predict_proba(x_test)[:, 1]
    auc_roc = roc_auc_score(y_test, y_prob_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob_pred)
    plt.figure(figsize=(12, 7))
    plt.plot(fpr, tpr, color="green", label=r'trained model (AUC=%.2f %%)'%(100*auc_roc))
    plt.plot([0, 1], [0, 1],  'r--',label='no skill')
    plt.title('ROC Curve for trained model \n (based on the test/hold-out set)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.fill_between(fpr, tpr, 0, facecolor='green', alpha=1)
    plt.grid('on')
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.show()
    return auc_roc


def calibrate(x_test: pd.DataFrame,
                     y_test: pd.DataFrame,
                     trained_model,
                     nbins: int=10):
    ypos_prob_pred=trained_model.predict_proba(x_test)[:, 1]
    prob_true, prob_pred = calibration_curve(y_test, ypos_prob_pred,
                                 n_bins=nbins, normalize=True)
    plt.figure(figsize=(12, 7))
    # plot perfectly calibrated
    plt.plot([0, 1], [0, 1], linestyle='--', label='perfectly calibrated' )
    # plot model reliability
    plt.plot(prob_pred, prob_true, marker='.', label='trained model')
    plt.title('Calibration for trained model \n (based on the test/hold-out set)')
    plt.xlabel('predicted probability of positive')
    plt.ylabel('actual probability of positive')
    plt.grid('on')
    plt.legend()
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.show()
    return prob_true, prob_pred


def classification_eval(xtest: pd.DataFrame,
                       ytest: pd.DataFrame, trainedModel,
                       bins: int=10):
    auc_roc = discrim_power(x_test=xtest, y_test=ytest,
                         trained_model=trainedModel)
    prob_true, prob_pred = calibrate(x_test=xtest, y_test=ytest,
                         trained_model=trainedModel, nbins=bins)
    return auc_roc, prob_true, prob_pred
