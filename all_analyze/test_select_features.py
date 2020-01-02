import pandas as pd
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_features, select_features
from sklearn.feature_selection import SelectFromModel, VarianceThreshold, SelectKBest, chi2
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.externals import joblib
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def load_data(name='all_analyze/data/all_in_one_data.csv'):
    df = pd.read_csv(name)
    Y = np.array(df['type'])
    del df['type']
    X = df
    print(X)

    train_X, test_X, train_y, test_y = train_test_split(X,
                                                        Y,
                                                        test_size=0.3,
                                                        random_state=0)
    print(train_X)
    print(train_y)
    train_X = train_X.reset_index(drop=True)
    train_X.to_csv('all_analyze/data/train_x.csv', index=False)
    pd.DataFrame(train_y).to_csv('all_analyze/data/train_y.csv', index=False, header=0)
    test_X = test_X.reset_index(drop=True)
    test_X.to_csv('all_analyze/data/test_x.csv', index=False)
    pd.DataFrame(test_y).to_csv('all_analyze/data/test_y.csv', index=False, header=0)

    return train_X, train_y


# 从文件读取feature,在已经保存全部特征的情况下使用
def _select_features(extracted_features, y):
    print(extracted_features)
    print('select start...')

    # 选取较相关的特征
    # 可选属性 fdr_level = 0.05 ?
    features_filtered = select_features(extracted_features, y, n_jobs=1, fdr_level=0.1, ml_task='classification')
    print(features_filtered)
    features_filtered.to_csv('test_classify/features/tsfresh_filteredFeatures.csv')
    print('select end')


# test sklearn SelectFromModel
def test_sklearn_SelectFromModel(extracted_features, y):
    cols = extracted_features.columns.values.tolist()
    print('select start...')

    y = np.array(np.array(y).tolist())
    extracted_features_arr = np.array(extracted_features)
    print(extracted_features)
    print(y)
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(extracted_features_arr, y)
    res = SelectFromModel(lsvc, prefit=True)
    features_filtered = res.transform(extracted_features_arr)
    cols = get_cols(cols, res.get_support())
    print(np.array(features_filtered))

    df = pd.DataFrame(features_filtered, columns=cols)
    df.to_csv('test_classify/features/test_sklearn_SelectFromModel.csv')


# test sklearn ExtraTreesClassifier
def test_sklearn_ExtraTreesClassifier(extracted_features, y):
    cols = extracted_features.columns.values.tolist()
    print('select start...')

    y = np.array(np.array(y).tolist())
    extracted_features_arr = np.array(extracted_features)
    print(extracted_features)
    print(y)
    clf = ExtraTreesClassifier(n_estimators=2, max_depth=3)
    clf = clf.fit(extracted_features_arr, y)
    res = SelectFromModel(clf, prefit=True)
    features_filtered = res.transform(extracted_features_arr)
    cols = get_cols(cols, res.get_support())
    print(np.array(features_filtered))

    df = pd.DataFrame(features_filtered, columns=cols)
    df.to_csv('test_classify/features/test_sklearn_ExtraTreesClassifier.csv')


# test
def get_cols(x, y):
    cols = []
    for i in range(len(y)):
        if y[i]:
            cols.append(x[i])
    return cols


# test sklearn VarianceThreshold
def test_sklearn_VarianceThreshold(extracted_features, y):
    cols = extracted_features.columns.values.tolist()
    print('select start...')

    y = np.array(np.array(y).tolist())
    extracted_features_arr = np.array(extracted_features)
    print(extracted_features)
    print(y)
    res = VarianceThreshold(threshold=(.6 * (1 - .6)))
    features_filtered = res.fit_transform(extracted_features_arr)
    cols = get_cols(cols, res.fit(extracted_features_arr).get_support())
    print(np.array(features_filtered))

    df = pd.DataFrame(features_filtered, columns=cols)
    df.to_csv('test_classify/features/test_sklearn_VarianceThreshold.csv')


def test_select_features_VarianceThreshold(y,
                                           extracted_features_name='test_classify/features/test_sklearn_VarianceThreshold.csv'):
    # 全部特征
    extracted_features = pd.read_csv(extracted_features_name)
    del extracted_features['Unnamed: 0']

    print(extracted_features)
    print('select start...')

    # 选取较相关的特征
    # 可选属性 fdr_level = 0.05 ?
    features_filtered = select_features(extracted_features, y, n_jobs=1, fdr_level=0.04, ml_task='classification')
    print(features_filtered)
    features_filtered.to_csv('test_classify/features/select_features_VarianceThreshold.csv')
    print('select end')


def start():
    print('start ...')
    extracted_features, y = load_data()

    print('filter')
    _select_features(extracted_features, y)

    print('linear')
    test_sklearn_SelectFromModel(extracted_features, y)

    print('tree')
    test_sklearn_ExtraTreesClassifier(extracted_features, y)

    print('varianceThreshold')
    test_sklearn_VarianceThreshold(extracted_features, y)
    test_select_features_VarianceThreshold(y)


start()
