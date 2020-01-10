import numpy as np
import pandas as pd
import os
import CONST

"""
simple tools
"""


def read_file(path=os.path.join(CONST.all_features_path, 'all_in_one_data.csv')):
    '''
    get x,y to classify
    :param path: 所有特征的csv
    :return: 处理好的dataframe
    '''

    df = pd.read_csv(path)
    cols = df.columns.values.tolist()
    for x in cols:
        if str(x).find('hjorth') < 0:
            if str(x).find('complexity') >= 0 or str(x).find('morbidity') >= 0:
                del df[x]
    y = df['type']
    del df['type']
    print(df)

    return df, np.array(y)


def handle_file(df, name, band):
    """

    :param df:
    :param name:
    :param band:
    :return:
    """
    cols = df.columns.values.tolist()
    for x in cols:
        if str(x).find(name) < 0 or str(x).find(band) < 0:
            del df[x]

    return df
