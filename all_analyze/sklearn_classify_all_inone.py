import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

'''
0.3 测试 0.7训练
'''


file_name = ['all_analyze/select_features/select_features_VarianceThreshold.csv',
             'all_analyze/select_features/test_sklearn_ExtraTreesClassifier.csv',
             'all_analyze/select_features/test_sklearn_SelectFromModel.csv',
             'all_analyze/select_features/tsfresh_filteredFeatures.csv']


def load_data(name):
    train_x = pd.read_csv(name)
    train_y = np.loadtxt('all_analyze/data/train_y.csv')

    del train_x['Unnamed: 0']
    train_cols = train_x.columns.values.tolist()

    train_x = np.array(train_x)
    test_x = pd.read_csv('all_analyze/data/test_x.csv')
    test_cols = test_x.columns.values.tolist()

    for x in test_cols:
        if x not in train_cols:
            del test_x[x]
    test_x = test_x.loc[:, train_cols]
    if name.find('ExtraTreesClassifier')>=0:
        print(test_x)
    test_x = np.array(test_x)
    test_y = np.loadtxt('all_analyze/data/test_y.csv')

    print(len(test_x))
    print(len(test_y))
    return train_x, train_y, test_x, test_y


# 贝叶斯
def naive_bayes_GaussianNB(x_train, x_test, y_train, y_test):
    # 将数据缩放到（0，1）
    scaling = MinMaxScaler(feature_range=(0, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)

    clf = GaussianNB()
    clf.fit(x_train, y_train)

    res = []
    expected = y_train
    predicted = clf.predict(x_train)
    res.append([precision_score(expected, predicted), recall_score(expected, predicted),
                accuracy_score(expected, predicted)])
    expected = y_test
    predicted = clf.predict(x_test)
    res.append([precision_score(expected, predicted), recall_score(expected, predicted),
                accuracy_score(expected, predicted)])

    return res


# 决策树
def decide_tree(x_train, x_test, y_train, y_test):
    # 将数据缩放到（0，1）

    scaling = MinMaxScaler(feature_range=(0, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)

    clf = tree.DecisionTreeClassifier(class_weight='balanced')
    clf = clf.fit(x_train, y_train.ravel())

    res = []
    expected = y_train
    predicted = clf.predict(x_train)
    res.append([precision_score(expected, predicted), recall_score(expected, predicted),
                accuracy_score(expected, predicted)])
    expected = y_test
    predicted = clf.predict(x_test)
    res.append([precision_score(expected, predicted), recall_score(expected, predicted),
                accuracy_score(expected, predicted)])

    return res


# svm
def linear_svm(x_train, x_test, y_train, y_test):
    # 将数据缩放到（0，1）
    scaling = MinMaxScaler(feature_range=(0, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)

    clf = svm.LinearSVC(penalty='l2', class_weight='balanced', loss='hinge')
    clf.fit(x_train, y_train)

    res = []
    expected = y_train
    predicted = clf.predict(x_train)
    res.append([precision_score(expected, predicted), recall_score(expected, predicted),
                accuracy_score(expected, predicted)])
    expected = y_test
    predicted = clf.predict(x_test)
    res.append([precision_score(expected, predicted), recall_score(expected, predicted),
                accuracy_score(expected, predicted)])

    return res


# knn
def k_n_n(x_train, x_test, y_train, y_test):
    # 将数据缩放到（0，1）
    scaling = MinMaxScaler(feature_range=(0, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)

    clf = KNeighborsClassifier()
    clf.fit(x_train, y_train)

    res = []
    expected = y_train
    predicted = clf.predict(x_train)
    res.append([precision_score(expected, predicted), recall_score(expected, predicted),
                accuracy_score(expected, predicted)])
    expected = y_test
    predicted = clf.predict(x_test)
    res.append([precision_score(expected, predicted), recall_score(expected, predicted),
                accuracy_score(expected, predicted)])

    return res


# 随机森林
def random_forest(x_train, x_test, y_train, y_test):
    # 将数据缩放到（0，1）
    scaling = MinMaxScaler(feature_range=(0, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)

    clf = RandomForestClassifier(n_estimators=100, max_depth=30, class_weight='balanced')
    clf.fit(x_train, y_train)

    res = []
    expected = y_train
    predicted = clf.predict(x_train)
    res.append([precision_score(expected, predicted), recall_score(expected, predicted),
                accuracy_score(expected, predicted)])
    expected = y_test
    predicted = clf.predict(x_test)
    res.append([precision_score(expected, predicted), recall_score(expected, predicted),
                accuracy_score(expected, predicted)])

    return res


def add_list_range(root,list1):
    for x in list1:
        root.append(x)
    return root


# 入口函数，分别计算十折交叉验证在几种模型下的精确率，召回率和准确率，并存成csv文件
def k_cv_3(name):
    colums = ['svm_precision', 'svm_recall', 'svm_accuracy', 'knn_precision', 'knn_recall', 'knn_accuracy',
              'tree_precision', 'tree_recall', 'tree_accuracy',
              'bayes_precision', 'bayes_recall', 'bayes_accuracy', 'forest_precision', 'forest_recall',
              'forest_accuracy']
    acc_pd = pd.DataFrame(columns=colums)

    x_train, y_train, x_test, y_test = load_data(name)

    temp = []
    temp1 = []

    print('svm:')
    res = linear_svm(x_train, x_test, y_train, y_test)
    print(res)
    add_list_range(temp,res[0])
    add_list_range(temp1,res[1])

    print('knn:')
    res = k_n_n(x_train, x_test, y_train, y_test)
    print(res)
    add_list_range(temp,res[0])
    add_list_range(temp1,res[1])

    print('decide tree:')
    res = decide_tree(x_train, x_test, y_train, y_test)
    print(res)
    add_list_range(temp,res[0])
    add_list_range(temp1,res[1])

    print('native bayes:')
    res = naive_bayes_GaussianNB(x_train, x_test, y_train, y_test)
    print(res)
    add_list_range(temp,res[0])
    add_list_range(temp1,res[1])

    print('random forest:')
    res = random_forest(x_train, x_test, y_train, y_test)
    print(res)
    add_list_range(temp,res[0])
    add_list_range(temp1,res[1])

    acc_pd.loc[len(acc_pd)] = temp
    acc_pd.loc[len(acc_pd)] = temp1

    acc_pd.rename(index={'0':'train', '1':'test'},inplace=True)

    acc_pd.to_csv('all_analyze/classify/test_sklearn_classify_' + str(name).split('/')[-1][:-4] + '.csv')


for x in file_name:
    k_cv_3(x)
