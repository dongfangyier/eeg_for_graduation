import pandas as pd
import numpy as np
import pymrmr
from sklearn.model_selection import train_test_split
from all_analyze import get_dataframe

import CONST
import os

'''
mrmr select features
'''


# 把数据转化成想要的格式
def read_file(name='all_analyze/data/train_x.csv'):
    df, Y = get_dataframe.read_file()

    train_X, test_X, train_y, test_y = train_test_split(df,
                                                        Y,
                                                        test_size=0.3,
                                                        random_state=0)

    df = train_X
    type = np.array(train_y)

    df.insert(0, 'type', type)
    cols = df.columns.values.tolist()
    for x in cols:
        if str(x).find('delta') < 0 and x is not 'type':
            del df[x]
    df.to_csv(os.path.join(CONST.classify_mrmr_path, 'all_analyze/data/all_in_one_data_origionmrmr.csv'))
    return df


def mrmr():
    df = read_file()
    print(df)
    # 输出了前20个特征
    temp = pymrmr.mRMR(df, 'MIQ', 20)
    print(temp)


mrmr()
