import pandas as pd
import numpy as np
import math
import numpy as np
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
def desc_filter(data):
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
    x_new = SelectKBest(chi2, k).fit_transform(X, y)
    print(x_new.shape)
    return x_new


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


def transform_to_numeric(data):

    data_new = remove_redundant(data)
    col = list(data_new.columns.values)

    # Tranformação dos atributos
    for c in col:
        le = preprocessing.LabelEncoder()
        if not is_numeric_dtype(data_new[c]):
            vals = data_new[c].unique()
            le.fit(vals)
            data_new[c] = le.transform(data_new[c])

    return data_new


def apply_filters(data):

    data = nan_filter(data)
    print(len(data.columns))

    data = corr_filter(data, "SalePrice")
    print(len(data.columns))

    data = desc_filter(data)
    print(len(data.columns))

    return data


def main():

    data = pd.read_csv("train.csv")
    data = data.drop(columns='Id')
    print('Train shape:', data.shape)

    '''data = apply_filters(data)
    print(data.shape)
    print(data.columns)'''

    data_new = remove_redundant(data)
    data_new = transform_to_numeric(data_new)
    # print(data_new.head())

    data_x, data_y = wrapper_rfe(data_new)
    print(data_x.shape, data_y.shape)


if __name__ == '__main__':
    main()
