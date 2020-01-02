import pandas as pd
import numpy as np
import pymrmr

'''
mrmr特征选择
'''


# 把数据转化成想要的格式
def read_file(name='all_analyze/data/train_x.csv'):
    df = pd.read_csv(name)
    type = np.loadtxt('all_analyze/data/train_y.csv')

    df.insert(0, 'type', type)
    cols = df.columns.values.tolist()
    for x in cols:
        if str(x).find('delta') < 0 and x is not 'type':
            del df[x]
    df.to_csv('all_analyze/data/all_in_one_data_origionmrmr.csv')
    return df


def mrmr():
    df = read_file()
    print(df)
    # 输出了前20个特征
    temp = pymrmr.mRMR(df, 'MIQ', 20)
    print(temp)


mrmr()
