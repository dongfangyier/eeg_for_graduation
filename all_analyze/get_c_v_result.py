import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import CONST
import os
from imblearn.over_sampling import SMOTE

'''
tools for classify
'''

# record max accurary
forest_acc = 0
IS_SOMIT = False

# 贝叶斯
def naive_bayes_GaussianNB(x_train, x_test, y_train, y_test):
    if IS_SOMIT:
        smo = SMOTE()
        x_train, y_train = smo.fit_sample(x_train, y_train)


    # 将数据缩放到（0，1）
    scaling = MinMaxScaler(feature_range=(0, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)

    clf = GaussianNB()
    clf.fit(x_train, y_train)

    expected = y_test
    predicted = clf.predict(x_test)

    matrix = confusion_matrix(expected, predicted)
    matrix = np.array(matrix)
    Specificity = matrix[0][0] / (matrix[0][0] + matrix[0][1])
    return precision_score(expected, predicted), recall_score(expected, predicted), accuracy_score(expected,
                                                                                                   predicted), Specificity


# 决策树
def decide_tree(x_train, x_test, y_train, y_test):
    if IS_SOMIT:
        smo = SMOTE()
        x_train, y_train = smo.fit_sample(x_train, y_train)

    # 将数据缩放到（0，1）

    scaling = MinMaxScaler(feature_range=(0, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)

    clf = tree.DecisionTreeClassifier(class_weight='balanced')
    clf = clf.fit(x_train, y_train.ravel())

    expected = y_test
    predicted = clf.predict(x_test)

    matrix = confusion_matrix(expected, predicted)
    matrix = np.array(matrix)
    Specificity = matrix[0][0] / (matrix[0][0] + matrix[0][1])
    return precision_score(expected, predicted), recall_score(expected, predicted), accuracy_score(expected,
                                                                                                   predicted), Specificity


# svm
def linear_svm(x_train, x_test, y_train, y_test):
    if IS_SOMIT:
        smo = SMOTE()
        x_train, y_train = smo.fit_sample(x_train, y_train)

    # 将数据缩放到（0，1）
    scaling = MinMaxScaler(feature_range=(0, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)

    clf = svm.LinearSVC(penalty='l2', class_weight='balanced', loss='hinge')
    clf.fit(x_train, y_train)

    expected = y_test
    predicted = clf.predict(x_test)

    matrix = confusion_matrix(expected, predicted)
    matrix = np.array(matrix)
    Specificity = matrix[0][0] / (matrix[0][0] + matrix[0][1])
    return precision_score(expected, predicted), recall_score(expected, predicted), accuracy_score(expected,
                                                                                                   predicted), Specificity


# knn
def k_n_n(x_train, x_test, y_train, y_test):
    if IS_SOMIT:
        smo = SMOTE()
        x_train, y_train = smo.fit_sample(x_train, y_train)

    # 将数据缩放到（0，1）
    scaling = MinMaxScaler(feature_range=(0, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)

    clf = KNeighborsClassifier()
    clf.fit(x_train, y_train)

    expected = y_test
    predicted = clf.predict(x_test)

    matrix = confusion_matrix(expected, predicted)
    matrix = np.array(matrix)
    Specificity = matrix[0][0] / (matrix[0][0] + matrix[0][1])
    return precision_score(expected, predicted), recall_score(expected, predicted), accuracy_score(expected,
                                                                                                   predicted), Specificity


# 随机森林
def random_forest(x_train, x_test, y_train, y_test):
    if IS_SOMIT:
        smo = SMOTE()
        x_train, y_train = smo.fit_sample(x_train, y_train)

    # 将数据缩放到（0，1）
    scaling = MinMaxScaler(feature_range=(0, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)

    clf = RandomForestClassifier(n_estimators=100, max_depth=30, class_weight='balanced')
    clf.fit(x_train, y_train)

    expected = y_test
    predicted = clf.predict(x_test)

    matrix = confusion_matrix(expected, predicted)
    Specificity = matrix[0][0] / (matrix[0][0] + matrix[0][1])

    #  model
    global forest_acc
    if accuracy_score(expected, predicted) > forest_acc:
        forest_acc = accuracy_score(expected, predicted)
        joblib.dump(clf, os.path.join(CONST.classify_model_path, 'tree_model_' + str(forest_acc) + '.pkl'))
        joblib.dump(scaling, os.path.join(CONST.classify_model_path, 'tree_scale_' + str(forest_acc) + '.pkl'))

    return precision_score(expected, predicted), recall_score(expected, predicted), accuracy_score(expected,
                                                                                                   predicted), Specificity


# 逻辑回归
def logistic_regression(x_train, x_test, y_train, y_test):
    if IS_SOMIT:
        smo = SMOTE()
        x_train, y_train = smo.fit_sample(x_train, y_train)

    # 将数据缩放到（0，1）
    scaling = MinMaxScaler(feature_range=(0, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)

    clf = LogisticRegression(class_weight='balanced')
    clf.fit(x_train, y_train)

    expected = y_test
    predicted = clf.predict(x_test)

    matrix = confusion_matrix(expected, predicted)
    matrix = np.array(matrix)
    Specificity = matrix[0][0] / (matrix[0][0] + matrix[0][1])
    return precision_score(expected, predicted), recall_score(expected, predicted), accuracy_score(expected,
                                                                                                   predicted), Specificity


# 入口函数，分别计算十折交叉验证在几种模型下的精确率，召回率和准确率，并存成csv文件
def k_cv_3(name, random, x, y):
    colums = ['svm_precision', 'svm_recall', 'svm_accuracy', 'knn_precision', 'knn_recall', 'knn_accuracy',
              'tree_precision', 'tree_recall', 'tree_accuracy',
              'bayes_precision', 'bayes_recall', 'bayes_accuracy', 'forest_precision', 'forest_recall',
              'forest_accuracy', 'logistic_precision', 'logistic_recall',
              'logistic_accuracy']

    colums_recall = ['svm_recall', 'knn_recall', 'tree_recall',
                     'bayes_recall', 'forest_recall', 'logistic_recall', ]

    colums_accuracy = ['svm_accuracy', 'knn_accuracy', 'tree_accuracy', 'bayes_accuracy',
                       'forest_accuracy', 'logistic_accuracy']

    colums_precision = ['svm_precision', 'knn_precision', 'tree_precision', 'bayes_precision',
                          'forest_precision', 'logistic_precision']

    acc_pd = pd.DataFrame(columns=colums)
    acc_rec = pd.DataFrame(columns=colums_recall)
    acc_acc = pd.DataFrame(columns=colums_accuracy)
    acc_pre = pd.DataFrame(columns=colums_precision)

    kf = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(x):
        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]
        temp = []
        temp_acc = []
        temp_rec = []
        temp_pre = []

        print('svm:')
        precision, recall, accuracy, spe = linear_svm(x_train, x_test, y_train, y_test)
        temp.append(precision)
        temp.append(recall)
        temp.append(accuracy)

        temp_acc.append(accuracy)
        temp_rec.append(recall)
        temp_pre.append(precision)

        print('knn:')
        precision, recall, accuracy, spe = k_n_n(x_train, x_test, y_train, y_test)
        temp.append(precision)
        temp.append(recall)
        temp.append(accuracy)

        temp_acc.append(accuracy)
        temp_rec.append(recall)
        temp_pre.append(precision)

        print('decide tree:')
        precision, recall, accuracy, spe = decide_tree(x_train, x_test, y_train, y_test)
        temp.append(precision)
        temp.append(recall)
        temp.append(accuracy)

        temp_acc.append(accuracy)
        temp_rec.append(recall)
        temp_pre.append(precision)

        print('native bayes:')
        precision, recall, accuracy, spe = naive_bayes_GaussianNB(x_train, x_test, y_train, y_test)
        temp.append(precision)
        temp.append(recall)
        temp.append(accuracy)

        temp_acc.append(accuracy)
        temp_rec.append(recall)
        temp_pre.append(precision)

        print('random forest:')
        precision, recall, accuracy, spe = random_forest(x_train, x_test, y_train, y_test)
        temp.append(precision)
        temp.append(recall)
        temp.append(accuracy)

        temp_acc.append(accuracy)
        temp_rec.append(recall)
        temp_pre.append(precision)

        print('logistic regression:')
        precision, recall, accuracy, spe = logistic_regression(x_train, x_test, y_train, y_test)
        temp.append(precision)
        temp.append(recall)
        temp.append(accuracy)

        temp_acc.append(accuracy)
        temp_rec.append(recall)
        temp_pre.append(precision)

        acc_pd.loc[len(acc_pd)] = temp
        acc_rec.loc[len(acc_rec)] = temp_rec
        acc_acc.loc[len(acc_acc)] = temp_acc
        acc_pre.loc[len(acc_pre)] = temp_pre

    acc_pd.loc['mean'] = acc_pd.mean()
    # ifdebug
    acc_pd.to_csv(os.path.join(CONST.classify_detail_path,
                               'classify_' + str(name) + '_c_k_' + str(random) + '.csv'))
    return acc_pd.mean(), acc_rec.mean(), acc_acc.mean(), acc_pre.mean()


colums11 = ['svm_precision', 'svm_recall', 'svm_accuracy', 'knn_precision', 'knn_recall', 'knn_accuracy',
            'tree_precision', 'tree_recall', 'tree_accuracy',
            'bayes_precision', 'bayes_recall', 'bayes_accuracy', 'forest_precision', 'forest_recall',
            'forest_accuracy', 'logistic_precision', 'logistic_recall',
            'logistic_accuracy']


def get_res(feature, band, x, y):
    colums_recall = ['svm_recall', 'knn_recall', 'tree_recall',
                     'bayes_recall', 'forest_recall', 'logistic_recall', ]

    colums_accuracy = ['svm_accuracy', 'knn_accuracy', 'tree_accuracy', 'bayes_accuracy',
                       'forest_accuracy', 'logistic_accuracy']
    colums_precision = ['svm_precision', 'knn_precision', 'tree_precision', 'bayes_precision',
                        'forest_precision', 'logistic_precision']
    acc_rec = pd.DataFrame(columns=colums_recall)
    acc_acc = pd.DataFrame(columns=colums_accuracy)
    acc_pre = pd.DataFrame(columns=colums_precision)
    res = pd.DataFrame(columns=colums11)
    for i in range(10):
        mean, rec, acc, pre = k_cv_3(feature, i, x, y)
        res.loc[len(res)] = mean
        acc_rec.loc[len(acc_rec)] = rec
        acc_acc.loc[len(acc_acc)] = acc
        acc_pre.loc[len(acc_pre)] = pre

    res.loc['mean'] = res.mean()
    # ifdebug
    res.to_csv(os.path.join(CONST.classify_detail_path, 'classify_' + str(feature) + '_' + str(band) + '_res_mean.csv'))

    return acc_rec.mean(), acc_acc.mean(), acc_pre.mean()
