import pandas as pd
import numpy as np
import math
import seaborn as sns
from sklearn import preprocessing
from scipy import stats
from pandas.api.types import is_numeric_dtype
from scipy.stats import mode
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import clone

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

remov = {}
remov["corr_filter"] = []
remov["nan_filter"] = []
remov["desc_filter"] = []


# Filter: NaN values
def nan_filter(data):
    global remov

    nan_attr = {}
    sum_attr = data.isnull().sum()

    key = 0
    for i in range(sum_attr.size):

        if (sum_attr.iloc[i] > 0):
            nan_attr[key] = [sum_attr.index[i], sum_attr.iloc[i]]
            key = key + 1

    for i in range(len(nan_attr)):

        if (nan_attr[i][1] >= 0.75 * data[nan_attr[i][0]].value_counts(dropna=False).sum()):
            data = data.drop(columns=nan_attr[i][0])
            remov["nan_filter"].append(nan_attr[i][0])
        elif (data[nan_attr[i][0]].dtype == 'object'):
            data[nan_attr[i][0]] = data[nan_attr[i][0]].fillna(data[nan_attr[i][0]].mode().iloc[0])
        else:
            data[nan_attr[i][0]] = data[nan_attr[i][0]].fillna(data[nan_attr[i][0]].mean())

    return data


# Filter: minimun correlation with target
def corr_filter(data, target):
    global remov
    cor = data.corr()
    cor_target = abs(cor[target])
    irrelevant_features = cor_target[cor_target < 0.6]
    data = data.drop(columns=irrelevant_features.index)
    remov["corr_filter"].append(irrelevant_features.index)
    return data


# Filter: unbalanced classes
def unbalanced_filter(data):
    global remov
    objects = []
    classes_attr = {}

    for i in range(len(data.columns)):
        if (data[data.iloc[0].index[i]].dtype == "object"):
            objects.append(data.iloc[0].index[i])

    for i in range(len(objects)):
        class_size = data[objects[i]].value_counts().size
        classes_attr[i] = []
        for j in range(class_size):
            classes_attr[i].append(data[objects[i]].value_counts()[j])

    for i in range(len(classes_attr)):
        if (sum(classes_attr[i][1:]) < classes_attr[i][0] / 2.5):
            remov["desc_filter"].append(objects[i])
            data = data.drop(columns=objects[i])

    return data


def wrapper_rfe(data_new):
    feature_names = list(data_new.columns.values)
    # seleção de atributos
    X = data_new
    X = X.drop('SalePrice', axis=1)
    y = data_new['SalePrice']

    regr = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=50)
    sel = RFE(regr, step=1)
    sel = sel.fit(X, y)
    attr = sorted(zip(map(lambda x: round(x, 4), sel.ranking_), feature_names))

    # validação
    i = 0
    atrib_sel = ['SalePrice']
    while attr[i][0] <= 20:
        atrib_sel.append(attr[i][1])
        i = i + 1

    data_cor = data_new[atrib_sel]
    print(data_cor.corr().sort_values('SalePrice', ascending=False).index)
    return data_cor, y


def wrapper_subconj_kbest(data, k=20):
    X = data
    X = X.drop('SalePrice', axis=1)
    y = data['SalePrice']
    selected_features = SelectKBest(chi2, k).fit(X, y).get_support()
    return selected_features, y


def remove_redundant(data):
    # Eliminando valores redundantes - data.drop_duplicates(inplace=True)
    ausentes = data.isnull()
    # Verificando colunas com NaN
    ausentes = pd.DataFrame(ausentes)
    col = list(data.columns.values)
    data_new = data
    for c in col:
        if len(data_new[c][data_new[c].isnull()]) != 0:
            if is_numeric_dtype(data_new[c]):
                mean = data_new[c].mean()
                data_new[c].fillna(mean, inplace=True)
            else:
                m = data_new[c].mode()[0]
                data_new[c].fillna(m, inplace=True)

    return data_new


def transform_to_numeric(data, should_remove_redundant=False):
    if should_remove_redundant:
        data = remove_redundant(data)

    col = list(data.columns.values)

    # Tranformação dos atributos
    for c in col:
        le = preprocessing.LabelEncoder()
        if not is_numeric_dtype(data[c]):
            vals = data[c].unique()
            le.fit(vals)
            data[c] = le.transform(data[c])

    return data


def apply_filters(data, nan=False, corr=False, unbalanced=False, is_training_data=True):
    if nan:
        print("Nan filter...")
        data = nan_filter(data)
        print(len(data.columns))

    if corr:
        if is_training_data:
            data = corr_filter(data, "SalePrice")
            print(len(data.columns))

    if unbalanced:
        data = unbalanced_filter(data)
        print(len(data.columns))

    return data


def run_model_on_filters(model, data, test, nan_filter=False, corr_filter=False,
                         unbalanced_filter=False):
    data = apply_filters(data, nan=nan_filter, corr=corr_filter, unbalanced=unbalanced_filter)
    test = apply_filters(test, nan=nan_filter, corr=corr_filter, unbalanced=unbalanced_filter,
                         is_training_data=False)

    for column in test.columns:
        if column not in data.columns:
            test = test.drop(column, axis=1)

    print(test.shape, data.shape)

    data = transform_to_numeric(data, True)  # put True for corr filter, False otherwise
    test = transform_to_numeric(test, True)

    data_y = data['SalePrice']
    data_x = data.drop(['SalePrice'], axis=1)

    model.fit(data_x, data_y)

    predictions = model.predict(test)

    return predictions


def run_model_on_rfe_wrapper(model, data, test):
    data = remove_redundant(data)
    data = transform_to_numeric(data)

    test = remove_redundant(test)
    test = transform_to_numeric(test)

    data_x, data_y = wrapper_rfe(data)

    print('RFE Wrapper:')

    for column in test.columns:
        if column not in data_x.columns:
            test = test.drop(column, axis=1)

    data_x = data_x.drop(['SalePrice'], axis=1)

    print(data_x.shape, test.shape)

    model.fit(data_x, data_y)

    predictions = model.predict(test)

    return predictions


def run_model_on_wrapper_subconj_kbest(model, data, test):
    data = remove_redundant(data)
    data = transform_to_numeric(data)

    test = remove_redundant(test)
    test = transform_to_numeric(test)

    print('KBest Wrapper:')

    features, data_y = wrapper_subconj_kbest(data)
    features = list(features)
    print(features)

    to_drop = []

    for k in range(len(features)):
        if not features[k]:
            to_drop.append(k)

    data = data.drop(data.columns[to_drop], axis=1)

    data = data.drop(['SalePrice'], axis=1)

    print(data.shape)

    model.fit(data, data_y)

    for column in test.columns:
        if column not in data.columns:
            test = test.drop(column, axis=1)

    predictions = model.predict(test)

    return predictions


def main():
    data = pd.read_csv("train.csv")
    data_copy = data

    test = pd.read_csv('test.csv')
    test_ids = test['Id']

    data = data.drop(columns='Id')
    print('Train shape:', data.shape)

    # prediction model:
    model = GradientBoostingRegressor(
        n_estimators=4864,
        learning_rate=0.05,
        max_depth=4,
        max_features='log2',
        min_samples_leaf=15,
        min_samples_split=10,
        loss='huber',
        random_state=5)

    # NaN Filter
    predictions = run_model_on_filters(clone(model), data, test, nan_filter=True)
    submission = pd.DataFrame({'Id': test_ids, 'Saleprice': predictions})
    submission.to_csv('submissions/nan_filter_submission.csv', index=False)
    print('Nan Filter - Submission file successfully created!')

    # Correlation filter
    predictions = run_model_on_filters(clone(model), data, test, corr_filter=True)
    submission = pd.DataFrame({'Id': test_ids, 'Saleprice': predictions})
    submission.to_csv('submissions/corr_filter_submission.csv', index=False)
    print('Correlation Filter - Submission file successfully created!')

    # Unbalanced filter
    predictions = run_model_on_filters(clone(model), data, test, unbalanced_filter=True)
    submission = pd.DataFrame({'Id': test_ids, 'Saleprice': predictions})
    submission.to_csv('submissions/unbalanced_filter_submission.csv', index=False)
    print('Unbalanced Filter - Submission file successfully created!')

    # Wrapper rfe
    predictions = run_model_on_rfe_wrapper(clone(model), data, test)
    submission = pd.DataFrame({'Id': test_ids, 'Saleprice': predictions})
    submission.to_csv('submissions/rfe_wrapper_submission.csv', index=False)
    print('RFE Wrapper: - Submission file successfully created!')

    # Wrapper KBest
    predictions = run_model_on_wrapper_subconj_kbest(clone(model), data, test)
    submission = pd.DataFrame({'Id': test_ids, 'Saleprice': predictions})
    submission.to_csv('submissions/KBest_wrapper_submission.csv', index=False)
    print('KBest Wrapper: - Submission file successfully created!')


if __name__ == '__main__':
    main()
