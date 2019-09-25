import os
import zipfile
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

def eval_recorder(params, tep, teprob, trp, trprob, scores, fi, name, mode='local'):
    scores = pd.DataFrame(scores, columns=['score'])
    mean_score = round(scores['score'].mean(), 6)
    std_score = round(scores['score'].std(), 4)
    name_score = str(mean_score) + '_' + str(std_score)
    if mode == 'local':
        if not os.path.isdir('results/' + name + '_' + name_score):
            os.mkdir('results/' + name + '_' + name_score)
        base_path = 'results/' + name + '_' + name_score + '/'
    if mode == 'kaggle':
        os.chdir(r'/kaggle/working')
        base_path = '/kaggle/working/'
        if not os.path.isdir(base_path + '/' + name + '_' + name_score):
            os.mkdir(name + '_' + name_score)
        base_path = base_path + '/' + name + '_' + name_score + '/'
    pd.Series(params).to_csv(base_path + 'params.csv', index=False)
    pd.DataFrame(tep).to_csv(base_path + 'test_predictions.csv', index=False)
    pd.DataFrame(teprob).to_csv(base_path + 'test_probablity.csv', index=False)
    if trp is not None:
        pd.DataFrame({'predition': trp, 'probablity': trprob}).to_csv(base_path + 'train_results.csv')
    pd.DataFrame(fi).to_csv(base_path + 'feature_importances.csv', index=False)
    if mode == 'kaggle':
        zipf = zipfile.ZipFile(name + '_' + name_score + '.zip', 'w', zipfile.ZIP_DEFLATED)
        zipdir(name + '_' + name_score +'/', zipf)
        zipf.close()
        return name + '_' + name_score + '.zip'
    
def split_eval(train, labels, x_val, y_val, test, clf, params, fit_params, name):
    scores = []
    feature_importances = np.zeros(len(train.columns))
    test_predictions = np.zeros(test.shape[0])
    test_probablity = np.zeros(test.shape[0])
    
    clf.fit(train, labels, eval_set=[(x_val, y_val)], **fit_params)
    if 'catboost' in name:
        scores.append(clf.best_score_['validation']['AUC'])
    if 'xgboost' in name:
        try:
            scores.append(clf.best_score)
        except:
            scores.append({'valid_0': {'auc': clf.evals_result()['validation_0']['auc'][-1]}})
    if 'lightgbm' in name:
        scores.append(clf.best_score_)
    test_predicts = clf.predict_proba(test)
    test_predictions = test_predicts[:, 1]
    test_probablity = test_predicts[:, 0]
    feature_importances = clf.feature_importances_
    print('-'*60)
    if 'lightgbm' in name:
        scores = [dict(s)['valid_0']['auc'] for s in scores]
    del clf
    filename = eval_recorder(params, test_predictions, test_probablity, None, None, scores, feature_importances, name, 'local')
    return test_predictions, test_probablity, None, None, scores, feature_importances, filename

def plot_feature_importances(fe, cols):
    fe = pd.DataFrame(fe, index=cols)
    if fe.shape[1] > 1:
        fe = fe.apply(sum, axis=1)
    else:
        fe = fe[0]
    fe.sort_values(ascending=False)[:20].plot(kind='bar')

def cv_eval(train, labels, test, clf, cv, params, fit_params, name):
    scores = []
    feature_importances = np.zeros((len(train.columns), cv.n_splits))
    train_predictions = np.zeros(train.shape[0])
    train_probablity = np.zeros(train.shape[0])
    test_predictions = np.zeros((test.shape[0], cv.n_splits))
    test_probablity = np.zeros((test.shape[0], cv.n_splits))
    for i, (train_index, val_index) in enumerate(cv.split(train, labels)):
        print(f'starting {i} split')
        x_train = train.iloc[train_index]
        y_train = labels[train_index]
        x_val = train.iloc[val_index]
        y_val = labels[val_index]
        clf.fit(x_train, y_train, eval_set=[(x_val, y_val)], **fit_params)
        if 'catboost' in name:
            scores.append(clf.best_score_['validation']['AUC'])
        if 'xgboost' in name:
            try:
                scores.append(clf.best_score)
            except:
                scores.append({'valid_0': {'auc': clf.evals_result()['validation_0']['auc'][-1]}})
        if 'lightgbm' in name:
            scores.append(clf.best_score_)
        val_predictions = clf.predict_proba(x_val)
        train_predictions[val_index] = val_predictions[:, 1]
        train_probablity[val_index] = val_predictions[:, 0]
        test_predicts = clf.predict_proba(test)
        test_predictions[:, i] = test_predicts[:, 1]
        test_probablity[:, i] = test_predicts[:, 0]
        feature_importances[:, i] = clf.feature_importances_
        print('-'*60)
    filename = eval_recorder(params, test_predictions, test_probablity, train_predictions, train_probablity, scores, feature_importances, name, 'local')
    return test_predictions, test_probablity, train_predictions, train_probablity, scores, feature_importances, filename

def eval_catboost(train, labels, test, cv, params, cat_features, name, eval_set=None, plot=False):
    clf = CatBoostClassifier(**params)
    fit_params = {
        'cat_features': cat_features,
        'plot': plot
    }
    if cv is not None:
        return cv_eval(train, labels, test, clf, cv, params, fit_params, 'catboost_' + name)
    return split_eval(train, labels, eval_set[0], eval_set[1], test, clf, params, fit_params, 'catboost_' + name)

def eval_xgboost(train, labels, test, cv, params, name, eval_set=None):
    clf = XGBClassifier(**params)
    fit_params = {
        'verbose':100, 
        'eval_metric':'auc',
        'early_stopping_rounds': 300
    }
    if cv is not None:
        return cv_eval(train, labels, test, clf, cv, params, fit_params, 'xgboost_' + name)
    return split_eval(train, labels, eval_set[0], eval_set[1], test, clf, params, fit_params, 'xgboost_' + name)

def eval_lightgbm(train, labels, test, cv, params, cat_features, name, eval_set=None):
    clf = LGBMClassifier(**params)
    fit_params = {
        'verbose': 100,
        'eval_metric': 'auc',
        #'categorical_feature':cat_features,        
        'early_stopping_rounds': 300
    }
    if cv is not None:
        return cv_eval(train, labels, test, clf, cv, params, fit_params, 'lightgbm_' + name)
    return split_eval(train, labels, eval_set[0], eval_set[1], test, clf, params, fit_params, 'lightgbm_' + name)