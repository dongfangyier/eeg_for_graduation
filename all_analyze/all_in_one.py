import numpy as np
import pandas as pd
'''
所有特征整理到1个csv
'''




control_file = ['data/c_features.csv', 'data/after_c_features_nonliear_dfa.csv',
                'data/after_c_features_nonliear_hfd.csv', 'data/after_c_features_nonliear_hjorth.csv',
                'data/after_c_features_nonliear_spectral_entropy.csv']
patient_file = ['data/p_features.csv', 'data/after_p_features_nonliear_dfa.csv',
                'data/after_p_features_nonliear_hfd.csv', 'data/after_p_features_nonliear_hjorth.csv',
                'data/after_p_features_nonliear_spectral_entropy.csv']

psd_control_name = ['psd_c_alpha1.csv', 'psd_c_alpha2.csv', 'psd_c_beta.csv', 'psd_c_delta.csv', 'psd_c_gamma.csv',
                    'psd_c_theta.csv']
psd_patient_name = ['psd_p_alpha1.csv', 'psd_p_alpha2.csv', 'psd_p_beta.csv', 'psd_p_delta.csv', 'psd_p_gamma.csv',
                    'psd_p_theta.csv']
psd_root_path = '/home/rbai/psd/'


def read_file(control, patient):
    c_df = pd.read_csv(control)
    c_df['type'] = 0
    p_df = pd.read_csv(patient)
    p_df['type'] = 1
    df = pd.concat([c_df, p_df], axis=0)
    del df['AAeid']
    del df['Unnamed: 0']
    df = df.reset_index(drop=True)

    return df


def read_psd_file(control, patient):
    c_df = pd.read_csv(psd_root_path + control)
    c_df.drop([1, 17], inplace=True)

    p_df = pd.read_csv(psd_root_path + patient)
    df = pd.concat([c_df, p_df], axis=0)
    del df['id']
    del df['Unnamed: 0']
    del df['groupId']

    band = str(control).split('_')[2][:-4]
    temp = df.columns.values.tolist()
    cols = {}
    for x in temp:
        cols[x] = band + '_psd_' + x
    df.rename(columns=cols, inplace=True)
    df = df.reset_index(drop=True)
    return df


def get_psd():
    df = None
    for i in range(len(psd_control_name)):
        temp = read_psd_file(psd_control_name[i], psd_patient_name[i])
        if df is None:
            df = temp
        else:
            df = pd.concat([df, temp], axis=1)
    return df


def all_in_one():
    df = None
    for i in range(len(control_file)):
        temp = read_file(control_file[i], patient_file[i])
        if df is None:
            df = temp
        else:
            del df['type']
            df = pd.concat([df, temp], axis=1)
    psd_df = get_psd()
    print(psd_df)
    print(df)

    df = pd.concat([psd_df, df], axis=1)
    df.to_csv('all_analyze/data/all_in_one_data.csv', index=False)


all_in_one()
