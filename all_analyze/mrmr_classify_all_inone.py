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
五折交叉验证
'''


gamma_35 = ['C2_gamma_hjorth_complexity', 'PO8_gamma_skewness', 'Fp2_gamma_skewness', 'FT8_gamma_skewness',
            'FT10_gamma_skewness', 'F7_gamma_skewness', 'AF8_gamma_skewness', 'FPz_gamma_skewness',
            'POz_gamma_skewness', 'Fp1_gamma_skewness', 'P8_gamma_skewness', 'Fz_gamma_skewness',
            'C4_gamma_skewness', 'TP7_gamma_skewness', 'AF3_gamma_skewness', 'F8_gamma_hjorth_complexity',
            'O1_gamma_skewness', 'F2_gamma_skewness', 'TP10_gamma_skewness', 'C4_gamma_absolute_centroid',
            'Fz_gamma_kurtosis', 'C5_gamma_hjorth_complexity', 'AF4_gamma_skewness', 'Cz_gamma_skewness',
            'FT9_gamma_skewness', 'POz_gamma_hjorth_complexity', 'FC3_gamma_skewness', 'C6_gamma_kurtosis',
            'C6_gamma_hjorth_complexity', 'CPz_gamma_skewness', 'C1_gamma_hjorth_complexity',
            'FC6_gamma_hjorth_complexity', 'CP1_gamma_skewness', 'FC1_gamma_kurtosis',
            'FC1_gamma_hjorth_complexity']

gamma = ['C2_gamma_hjorth_complexity', 'PO8_gamma_skewness', 'Fp2_gamma_skewness', 'FT8_gamma_skewness',
         'FT10_gamma_skewness',
         'F7_gamma_skewness', 'AF8_gamma_skewness', 'FPz_gamma_skewness', 'POz_gamma_skewness', 'Fp1_gamma_skewness',
         'P8_gamma_skewness',
         'Fz_gamma_skewness', 'C4_gamma_skewness', 'TP7_gamma_skewness', 'AF3_gamma_skewness',
         'F8_gamma_hjorth_complexity',
         'O1_gamma_skewness', 'F2_gamma_skewness', 'TP10_gamma_skewness', 'C4_gamma_absolute_centroid']

alpha1 = ['P4_alpha1_hjorth_complexity', 'AF8_alpha1_skewness', 'Cz_alpha1_skewness', 'CP1_alpha1_skewness',
          'FT8_alpha1_skewness', 'Fp1_alpha1_skewness', 'C4_alpha1_skewness', 'FPz_alpha1_skewness',
          'AF3_alpha1_skewness', 'TP7_alpha1_skewness', 'FT9_alpha1_skewness', 'Fp2_alpha1_absolute_centroid',
          'FC4_alpha1_skewness', 'POz_alpha1_skewness', 'AF4_alpha1_skewness', 'FC3_alpha1_skewness',
          'FC1_alpha1_skewness', 'CPz_alpha1_skewness', 'FC2_alpha1_absolute_centroid', 'Fz_alpha1_skewness']

alpha2 = ['C1_alpha2_hjorth_complexity', 'FT8_alpha2_skewness', 'AF8_alpha2_skewness', 'Cz_alpha2_skewness',
          'FC3_alpha2_skewness', 'Fp1_alpha2_skewness', 'C4_alpha2_skewness', 'FPz_alpha2_skewness',
          'TP7_alpha2_skewness', 'CP1_alpha2_skewness', 'AF3_alpha2_skewness', 'AF4_alpha2_skewness',
          'FC1_alpha2_skewness', 'FT9_alpha2_skewness', 'POz_alpha2_skewness', 'CPz_alpha2_skewness',
          'TP9_alpha2_skewness', 'Pz_alpha2_skewness', 'FC1_alpha2_hjorth_complexity', 'F1_alpha2_skewness']

theta = ['PO7_theta_hjorth_complexity', 'C2_theta_absolute_centroid', 'TP7_theta_skewness', 'AF3_theta_skewness',
         'AF8_theta_skewness', 'TP10_theta_skewness', 'C4_theta_absolute_centroid', 'P8_theta_skewness',
         'Fz_theta_skewness', 'POz_theta_skewness', 'O1_theta_absolute_centroid', 'FPz_theta_skewness',
         'PO8_theta_skewness', 'F5_theta_hjorth_complexity', 'CP1_theta_skewness', 'FC1_theta_hjorth_complexity',
         'FT9_theta_skewness', 'AF3_theta_hjorth_complexity', 'CPz_theta_skewness', 'C5_theta_hjorth_complexity']

beta = ['Cz_beta_hjorth_complexity', 'AF8_beta_skewness', 'CP2_beta_skewness', 'FT8_beta_skewness', 'Cz_beta_skewness',
        'P8_beta_skewness', 'AF4_beta_skewness', 'TP7_beta_skewness', 'P4_beta_hjorth_complexity', 'Fz_beta_skewness',
        'O1_beta_skewness', 'POz_beta_skewness', 'C3_beta_hjorth_complexity', 'FT10_beta_skewness', 'FC3_beta_skewness',
        'PO8_beta_skewness', 'Pz_beta_hjorth_complexity', 'FC2_beta_skewness', 'Fp1_beta_skewness',
        'P8_beta_hjorth_complexity']

delta = ['FC6_delta_hjorth_complexity', 'TP8_delta_hjorth_complexity', 'FT8_delta_hjorth_complexity',
         'CP2_delta_hjorth_complexity', 'TP7_delta_hjorth_complexity', 'CP6_delta_hjorth_complexity',
         'T8_delta_hjorth_complexity', 'FC2_delta_hjorth_complexity', 'TP9_delta_hjorth_complexity',
         'T7_delta_hjorth_complexity', 'C5_delta_hjorth_complexity', 'FC4_delta_hjorth_complexity',
         'FC3_delta_hjorth_complexity', 'F3_delta_hjorth_complexity', 'P8_delta_hjorth_complexity',
         'P5_delta_hjorth_complexity', 'FT7_delta_hjorth_complexity', 'F4_delta_hjorth_complexity',
         'FC5_delta_hjorth_complexity', 'CP3_delta_hjorth_complexity']


# 把数据处理成想要的格式
def load_data(name='all_analyze/data/all_in_one_data.csv'):
    df = pd.read_csv(name)
    Y = np.array(df['type'])
    del df['type']
    cols = df.columns.values.tolist()
    for x in cols:
        if x not in gamma_35:
            del df[x]

    print(df)
    X = np.array(df)

    print(Y)

    return X, Y



# 贝叶斯
def naive_bayes_GaussianNB(x_train, x_test, y_train, y_test):
    # x_train = preprocessing.scale(x_train)
    # x_test = preprocessing.scale(x_test)

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
    # x_train = preprocessing.scale(x_train)
    # x_test = preprocessing.scale(x_test)

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
    # x_train=preprocessing.scale(x_train)
    # x_test=preprocessing.scale(x_test)

    # 将数据缩放到（0，1）
    scaling = MinMaxScaler(feature_range=(0, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)

    clf = svm.LinearSVC(penalty='l2', class_weight='balanced', loss='hinge')
    clf.fit(x_train, y_train)

    # expected = y_train
    # predicted = clf.predict(x_train)
    # print(metrics.classification_report(expected, predicted))

    expected = y_test
    predicted = clf.predict(x_test)
    print(metrics.classification_report(expected, predicted))

    return precision_score(expected, predicted), recall_score(expected, predicted), accuracy_score(expected, predicted)


# knn
def k_n_n(x_train, x_test, y_train, y_test):
    # x_train = preprocessing.scale(x_train)
    # x_test = preprocessing.scale(x_test)

    # 将数据缩放到（0，1）
    scaling = MinMaxScaler(feature_range=(0, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)

    clf = KNeighborsClassifier()
    clf.fit(x_train, y_train)

    # expected = y_train
    # predicted = clf.predict(x_train)
    # print(metrics.classification_report(expected, predicted))

    expected = y_test
    predicted = clf.predict(x_test)
    print(metrics.classification_report(expected, predicted))

    return precision_score(expected, predicted), recall_score(expected, predicted), accuracy_score(expected, predicted)


# 随机森林
def random_forest(x_train, x_test, y_train, y_test):
    # x_train = preprocessing.scale(x_train)
    # x_test = preprocessing.scale(x_test)

    # 将数据缩放到（0，1）
    scaling = MinMaxScaler(feature_range=(0, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)

    clf = RandomForestClassifier(n_estimators=100, max_depth=30, class_weight='balanced')
    clf.fit(x_train, y_train)

    expected = y_test
    predicted = clf.predict(x_test)
    print(metrics.classification_report(expected, predicted))

    return precision_score(expected, predicted), recall_score(expected, predicted), accuracy_score(expected, predicted)


# 入口函数，分别计算十折交叉验证在几种模型下的精确率，召回率和准确率，并存成csv文件
def k_cv_3():
    colums = ['svm_precision', 'svm_recall', 'svm_accuracy', 'knn_precision', 'knn_recall', 'knn_accuracy',
              'tree_precision', 'tree_recall', 'tree_accuracy',
              'bayes_precision', 'bayes_recall', 'bayes_accuracy', 'forest_precision', 'forest_recall',
              'forest_accuracy']
    acc_pd = pd.DataFrame(columns=colums)

    x, y = load_data()
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

        acc_pd.loc[len(acc_pd)] = temp

    acc_pd.loc['mean'] = acc_pd.mean()
    acc_pd.to_csv('all_analyze/classify/test_mrmr_gamma_35_classify_ck.csv')


k_cv_3()
