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
from sklearn.linear_model import LogisticRegression



'''
五折交叉验证
'''

file_names = ['all_analyze/select_features/select_features_VarianceThreshold.csv',
             'all_analyze/select_features/test_sklearn_ExtraTreesClassifier.csv',
             'all_analyze/select_features/test_sklearn_SelectFromModel.csv',
             'all_analyze/select_features/tsfresh_filteredFeatures.csv']

# file_names = ['test_classify/features/select_features_VarianceThreshold.csv',
#              'test_classify/features/test_sklearn_ExtraTreesClassifier.csv',
#              'test_classify/features/test_sklearn_SelectFromModel.csv',
#              'test_classify/features/tsfresh_filteredFeatures.csv']


def load_data(name, name1='all_analyze/data/all_in_one_data.csv'):
    df = pd.read_csv(name)
    del df['Unnamed: 0']
    train_cols = df.columns.values.tolist()

    X = pd.read_csv(name1)
    Y = X['type']
    del X['type']
    x_cols = X.columns.values.tolist()
    for x in x_cols:
        if x not in train_cols:
            del X[x]
    print(X)

    X = np.array(X)
    Y = np.array(Y)
    print(Y)

    return X, Y


# 贝叶斯
def naive_bayes_GaussianNB(x_train, x_test, y_train, y_test):
    # 将数据缩放到（0，1）
    scaling = MinMaxScaler(feature_range=(0, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)

    clf = GaussianNB()
    clf.fit(x_train, y_train)

    expected = y_test
    predicted = clf.predict(x_test)

    print(metrics.classification_report(expected, predicted))

    return precision_score(expected, predicted), recall_score(expected, predicted), accuracy_score(expected, predicted)


# 决策树
def decide_tree(x_train, x_test, y_train, y_test):
    # 将数据缩放到（0，1）

    scaling = MinMaxScaler(feature_range=(0, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)

    clf = tree.DecisionTreeClassifier(class_weight='balanced')
    clf = clf.fit(x_train, y_train.ravel())

    expected = y_test
    predicted = clf.predict(x_test)
    print(metrics.classification_report(expected, predicted))

    return precision_score(expected, predicted), recall_score(expected, predicted), accuracy_score(expected, predicted)


# svm
def linear_svm(x_train, x_test, y_train, y_test):
    # 将数据缩放到（0，1）
    scaling = MinMaxScaler(feature_range=(0, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)

    clf = svm.SVC(kernel='sigmoid', class_weight='balanced')
    clf.fit(x_train, y_train)

    expected = y_test
    predicted = clf.predict(x_test)
    print(metrics.classification_report(expected, predicted))

    return precision_score(expected, predicted), recall_score(expected, predicted), accuracy_score(expected, predicted)


# knn
def k_n_n(x_train, x_test, y_train, y_test):
    # 将数据缩放到（0，1）
    scaling = MinMaxScaler(feature_range=(0, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)

    clf = KNeighborsClassifier()
    clf.fit(x_train, y_train)

    expected = y_test
    predicted = clf.predict(x_test)
    print(metrics.classification_report(expected, predicted))

    return precision_score(expected, predicted), recall_score(expected, predicted), accuracy_score(expected, predicted)


# 随机森林
def random_forest(x_train, x_test, y_train, y_test):
    # 将数据缩放到（0，1）
    scaling = MinMaxScaler(feature_range=(0, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)

    clf = RandomForestClassifier(n_estimators=100, max_depth=30, class_weight='balanced')
    clf.fit(x_train, y_train)

    expected = y_test
    predicted = clf.predict(x_test)
    print(metrics.classification_report(expected, predicted))
    print(expected)
    print(predicted)

    return precision_score(expected, predicted), recall_score(expected, predicted), accuracy_score(expected, predicted)


def logistic_regression(x_train, x_test, y_train, y_test):
    # 将数据缩放到（0，1）
    scaling = MinMaxScaler(feature_range=(0, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)

    clf = LogisticRegression(class_weight='balanced')
    clf.fit(x_train, y_train)

    expected = y_test
    predicted = clf.predict(x_test)
    print(metrics.classification_report(expected, predicted))
    print(expected)
    print(predicted)

    return precision_score(expected, predicted), recall_score(expected, predicted), accuracy_score(expected, predicted)



# 入口函数，分别计算十折交叉验证在几种模型下的精确率，召回率和准确率，并存成csv文件
def k_cv_3(name, random):
    colums = ['svm_precision', 'svm_recall', 'svm_accuracy', 'knn_precision', 'knn_recall', 'knn_accuracy',
              'tree_precision', 'tree_recall', 'tree_accuracy',
              'bayes_precision', 'bayes_recall', 'bayes_accuracy', 'forest_precision', 'forest_recall',
              'forest_accuracy', 'logistic_precision', 'logistic_recall',
              'logistic_accuracy']
    acc_pd = pd.DataFrame(columns=colums)

    x, y = load_data(name)
    print(y)

    kf = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(x):
        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]
        temp = []

        print('svm:')
        precision, recall, accuracy = linear_svm(x_train, x_test, y_train, y_test)
        temp.append(precision)
        temp.append(recall)
        temp.append(accuracy)

        print('knn:')
        precision, recall, accuracy = k_n_n(x_train, x_test, y_train, y_test)
        temp.append(precision)
        temp.append(recall)
        temp.append(accuracy)

        print('decide tree:')
        precision, recall, accuracy = decide_tree(x_train, x_test, y_train, y_test)
        temp.append(precision)
        temp.append(recall)
        temp.append(accuracy)

        print('native bayes:')
        precision, recall, accuracy = naive_bayes_GaussianNB(x_train, x_test, y_train, y_test)
        temp.append(precision)
        temp.append(recall)
        temp.append(accuracy)

        print('random forest:')
        precision, recall, accuracy = random_forest(x_train, x_test, y_train, y_test)
        temp.append(precision)
        temp.append(recall)
        temp.append(accuracy)

        print('logistic regression:')
        precision, recall, accuracy = random_forest(x_train, x_test, y_train, y_test)
        temp.append(precision)
        temp.append(recall)
        temp.append(accuracy)

        acc_pd.loc[len(acc_pd)] = temp

    acc_pd.loc['mean'] = acc_pd.mean()
    acc_pd.to_csv('test_classify/result/classify_' + str(name).split('/')[2][:-4] + '_c_k_' + str(random) + '.csv')
    return acc_pd.mean()


colums11 = ['svm_precision', 'svm_recall', 'svm_accuracy', 'knn_precision', 'knn_recall', 'knn_accuracy',
          'tree_precision', 'tree_recall', 'tree_accuracy',
          'bayes_precision', 'bayes_recall', 'bayes_accuracy', 'forest_precision', 'forest_recall',
          'forest_accuracy', 'logistic_precision', 'logistic_recall',
          'logistic_accuracy']
for file_name in file_names:
    res = pd.DataFrame(columns=colums11)
    for i in range(10):
        mean = k_cv_3(file_name, i)
        res.loc[len(res)] = mean
    res.loc['mean'] = res.mean()
    res.to_csv('test_classify/result/classify'+ str(file_name).split('/')[2][:-4] +'_res_mean.csv')
