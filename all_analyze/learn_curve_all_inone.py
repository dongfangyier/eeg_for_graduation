import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import CONST
import os

'''
sklean ExtraTreesClassifier 选取特征的学习曲线

'''


def load_data(name=os.path.join(CONST.select_features_path, 'test_sklearn_ExtraTreesClassifier.csv'),
              name1=os.path.join(CONST.all_features_path, 'all_in_one_data.csv')):
    df = pd.read_csv(name)
    del df['Unnamed: 0']
    train_cols = df.columns.values.tolist()

    X = pd.read_csv(name1)
    print(X)
    Y = X['type']
    del X['type']
    x_cols = X.columns.values.tolist()
    for x in x_cols:
        if x not in train_cols:
            del X[x]
    X = np.array(X)
    Y = np.array(Y)
    print(Y)

    return X, Y


def get_score(x, y, clf):
    train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    train_score = []
    test_score = []
    # clf = svm.SVC(kernel='linear', C=1.3, decision_function_shape='ovr',class_weight='balanced')
    for i in train_sizes:
        temp_train = []
        temp_test = []
        for my_random in range(20):
            x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=my_random, train_size=i)
            scaling = MinMaxScaler(feature_range=(0, 1)).fit(x_train)
            x_train = scaling.transform(x_train)
            x_test = scaling.transform(x_test)

            clf.fit(x_train, y_train.ravel())
            temp_train.append(clf.score(x_train, y_train))
            temp_test.append(clf.score(x_test, y_test))
        train_score.append(temp_train)
        test_score.append(temp_test)
    return train_sizes, np.array(train_score), np.array(test_score)


def curve(clf, name):
    x, y = load_data()
    # clf = svm.SVC(kernel='linear', C=1.3, decision_function_shape='ovo')
    print(y)
    print(len(y))
    train_sizes, train_score, test_score = get_score(x, y, clf)
    # train_sizes, train_score, test_score = learning_curve(clf,x,y,train_sizes=[0.1,0.2,0.4,0.6,0.8],cv=None,scoring='accuracy')
    # train_error = 1 - np.mean(train_score, axis=1)
    # test_error = 1 - np.mean(test_score, axis=1)
    train_score = np.mean(train_score, axis=1)
    test_score = np.mean(test_score, axis=1)
    plt.plot(train_sizes, train_score, 'o-', color='r', label='training')
    plt.plot(train_sizes, test_score, 'o-', color='g', label='testing')
    plt.legend(loc='best')
    plt.xlabel('traing examples')
    plt.ylabel('accuracy')
    plt.title(name + '  learning curve')
    plt.show()
    plt.savefig(os.path.join(CONST.classify_image_path, name + '_forest.png'))
    plt.close()


def start():
    name = ['svm', 'bayes', 'decision tree', 'random forest', 'knn']
    clf = svm.LinearSVC(penalty='l2', class_weight='balanced', loss='hinge')
    curve(clf, name[0])
    clf = GaussianNB()
    curve(clf, name[1])
    clf = tree.DecisionTreeClassifier(class_weight='balanced')
    curve(clf, name[2])
    clf = RandomForestClassifier(n_estimators=100, max_depth=30, class_weight='balanced')
    curve(clf, name[3])
    clf = KNeighborsClassifier()
    curve(clf, name[4])


start()
