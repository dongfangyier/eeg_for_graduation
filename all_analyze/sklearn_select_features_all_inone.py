import pandas as pd
from tsfresh import extract_features, select_features
from sklearn.feature_selection import SelectFromModel, VarianceThreshold, SelectKBest, chi2
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import CONST
import os
import all_analyze.get_dataframe as get_dataframe


'''
features selected
'''


def load_data():
    """
    load all data and split it, only choose 30% data for select features
    :param name:
    :return:
    """
    df, Y = get_dataframe.read_file()

    train_X, test_X, train_y, test_y = train_test_split(df,
                                                        Y,
                                                        test_size=0.3,
                                                        random_state=0)
    print(train_X)
    print(train_y)
    train_X = train_X.reset_index(drop=True)

    # ifdebug
    # train_X.to_csv('all_analyze/data/train_x.csv', index=False)
    # pd.DataFrame(train_y).to_csv('all_analyze/data/train_y.csv', index=False, header=0)
    # test_X = test_X.reset_index(drop=True)
    # test_X.to_csv('all_analyze/data/test_x.csv', index=False)
    # pd.DataFrame(test_y).to_csv('all_analyze/data/test_y.csv', index=False, header=0)

    return train_X, train_y


# 从文件读取feature,在已经保存全部特征的情况下使用
def _select_features(extracted_features, y):
    """
    select features using tsfresh's function which is depending on fdr
    :param extracted_features:
    :param y:
    :return:
    """
    print(extracted_features)
    print('select start...')

    # 选取较相关的特征
    # 可选属性 fdr_level = 0.05 ?
    features_filtered = select_features(extracted_features, y, n_jobs=1, fdr_level=0.05, ml_task='classification')
    print(features_filtered)
    features_filtered.to_csv(os.path.join(CONST.select_features_path, "tsfresh_filteredFeatures.csv"))
    print('select end')


# test sklearn SelectFromModel
def test_sklearn_SelectFromModel(extracted_features, y):
    """
    select features using linear mode
    :param extracted_features:
    :param y:
    :return:
    """
    cols = extracted_features.columns.values.tolist()
    print('select start...')

    y = np.array(np.array(y).tolist())
    extracted_features_arr = np.array(extracted_features)
    print(extracted_features)
    print(y)
    lsvc = LinearSVC(C=1, penalty="l1", dual=False).fit(extracted_features_arr, y)
    res = SelectFromModel(lsvc, prefit=True)
    features_filtered = res.transform(extracted_features_arr)
    cols = get_cols(cols, res.get_support())
    print(np.array(features_filtered))

    df = pd.DataFrame(features_filtered, columns=cols)
    df.to_csv(os.path.join(CONST.select_features_path, "test_sklearn_SelectFromModel.csv"))


# test sklearn ExtraTreesClassifier
def test_sklearn_ExtraTreesClassifier(extracted_features, y):
    """
    select features using tree mode
    :param extracted_features:
    :param y:
    :return:
    """
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
    df.to_csv(os.path.join(CONST.select_features_path, "test_sklearn_ExtraTreesClassifier.csv"))


# test
def get_cols(x, y):
    """
    get columns depend on index
    :param x:
    :param y:
    :return:
    """
    cols = []
    for i in range(len(y)):
        if y[i]:
            cols.append(x[i])
    return cols


# test sklearn VarianceThreshold
def test_sklearn_VarianceThreshold(extracted_features, y):
    """
    remove low variance features(maybe 0.6 is suitable),this function only leave a temp result
    :param extracted_features:
    :param y:
    :return:
    """
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
    df.to_csv(os.path.join(CONST.select_features_path, "test_sklearn_VarianceThreshold.csv"))


def test_select_features_VarianceThreshold(y,
                                           extracted_features_name=os.path.join(CONST.select_features_path, "test_sklearn_VarianceThreshold.csv")):
    """
    select features using tsfresh's function which is depending on fdr
    this function use the result of test_sklearn_VarianceThreshold
    :param y:
    :param extracted_features_name:
    :return:
    """
    # 全部特征
    extracted_features = pd.read_csv(extracted_features_name)
    del extracted_features['Unnamed: 0']

    print(extracted_features)
    print('select start...')

    # 选取较相关的特征
    # 可选属性 fdr_level = 0.05 ?
    features_filtered = select_features(extracted_features, y, n_jobs=1, fdr_level=0.2, ml_task='classification')
    print(features_filtered)
    features_filtered.to_csv(os.path.join(CONST.select_features_path, "select_features_VarianceThreshold.csv"))
    print('select end')


def start():
    """
    main function in this script for select features
    :return:
    """
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


