import numpy as np
import pandas as pd
import all_analyze.get_c_v_result as get_c_v_result
import all_analyze.get_dataframe as get_dataframe
import CONST
import os


"""
classify main function
"""



file_names = ['test_sklearn_SelectFromModel.csv',
              'test_sklearn_ExtraTreesClassifier.csv',
              'tsfresh_filteredFeatures.csv',
              'select_features_VarianceThreshold.csv', ]

columns_rec = ['method', 'count', 'svm_recall', 'knn_recall', 'tree_recall',
               'bayes_recall', 'forest_recall', 'logistic_recall']

columns_acc = ['method', 'count', 'svm_accuracy', 'knn_accuracy', 'tree_accuracy', 'bayes_accuracy',
               'forest_accuracy', 'logistic_accuracy']

columns_pre = ['method', 'count', 'svm_precision', 'knn_precision', 'tree_precision', 'bayes_precision',
               'forest_precision', 'logistic_precision']

methods = ['linear', 'tree', 'tsfresh_select', 'VarianceThreshold_tsfresh_select']


def get_features(df, path):
    """
    get features which is selected
    :param df:
    :param path:
    :return:
    """
    df1 = pd.read_csv(path)
    del df1['Unnamed: 0']
    cols = df1.columns.values.tolist()
    x_cols = df.columns.values.tolist()
    for x in x_cols:
        if x not in cols:
            del df[x]
    return df


def start():
    df, y = get_dataframe.read_file()
    df_rec = pd.DataFrame(columns=columns_rec)
    df_acc = pd.DataFrame(columns=columns_acc)
    df_pre = pd.DataFrame(columns=columns_pre)

    '''
    all
    '''
    rec, acc,pre = get_c_v_result.get_res('select_features', 'test', np.array(df.copy()), y)
    temp = pd.Series({'method': 'all_features', 'count': str(len(np.array(df.copy())[0]))})
    temp = pd.concat([temp, rec])
    df_rec.loc[len(df_rec)] = temp
    print(df_rec)

    temp = pd.Series({'method': 'all_features', 'count': str(len(np.array(df.copy())[0]))})
    temp = pd.concat([temp, acc])
    df_acc.loc[len(df_acc)] = temp
    print(df_acc)

    temp = pd.Series({'method': 'all_features', 'count': str(len(np.array(df.copy())[0]))})
    temp = pd.concat([temp, pre])
    df_pre.loc[len(df_pre)] = temp
    print(df_pre)

    '''
    select features result
    '''

    for i in range(len(methods)):
        X = get_features(df.copy(), os.path.join(CONST.select_features_path, file_names[i]))
        rec, acc, pre = get_c_v_result.get_res('select_features', str(i), np.array(X.copy()), y)
        temp = pd.Series({'method': methods[i], 'count': len(np.array(X)[0])})
        temp = pd.concat([temp, rec])
        df_rec.loc[len(df_rec)] = temp
        print(df_rec)

        temp = pd.Series({'method': methods[i], 'count': len(np.array(X)[0])})
        temp = pd.concat([temp, acc])
        df_acc.loc[len(df_acc)] = temp
        print(df_acc)

        temp = pd.Series({'method': methods[i], 'count': len(np.array(X)[0])})
        temp = pd.concat([temp, pre])
        df_pre.loc[len(df_pre)] = temp
        print(df_pre)

    df_rec.to_csv(os.path.join(CONST.classify_path, "recall.csv"))
    df_acc.to_csv(os.path.join(CONST.classify_path, "accuracy.csv"))
    df_pre.to_csv(os.path.join(CONST.classify_path, "precision.csv"))


