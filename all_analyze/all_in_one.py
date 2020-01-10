import os
import CONST
import pandas as pd

'''
merge all features into one csv
'''

control_file = ['c_features.csv', 'c_features_nonliear_dfa.csv',
                'c_features_nonliear_hfd.csv', 'c_features_nonliear_hjorth.csv',
                'c_features_nonliear_spectral_entropy.csv']
patient_file = ['p_features.csv', 'p_features_nonliear_dfa.csv',
                'p_features_nonliear_hfd.csv', 'p_features_nonliear_hjorth.csv',
                'p_features_nonliear_spectral_entropy.csv']

psd_control_name = ['psd_c_alpha1.csv', 'psd_c_alpha2.csv', 'psd_c_beta.csv', 'psd_c_delta.csv', 'psd_c_gamma.csv',
                    'psd_c_theta.csv']
psd_patient_name = ['psd_p_alpha1.csv', 'psd_p_alpha2.csv', 'psd_p_beta.csv', 'psd_p_delta.csv', 'psd_p_gamma.csv',
                    'psd_p_theta.csv']



def concact_file(control, patient):
    """
    concat patient csv and control csv into one csv and add column named "type"
    :param control:
    :param patient:
    :return:
    """
    c_df = pd.read_csv(os.path.join(CONST.features_path, control))
    c_df['type'] = 0
    p_df = pd.read_csv(os.path.join(CONST.features_path, patient))
    p_df['type'] = 1
    df = pd.concat([c_df, p_df], axis=0)
    del df['Unnamed: 0']
    df = df.reset_index(drop=True)

    return df


def read_psd_file(control, patient):
    """
    handled psd file because psd is not as same as other features
    :param control:
    :param patient:
    :return:
    """
    c_df = pd.read_csv(os.path.join(CONST.features_path,  control))

    p_df = pd.read_csv(os.path.join(CONST.features_path,  patient))
    df = pd.concat([c_df, p_df], axis=0)
    del df['Unnamed: 0']
    del df['groupId']

    band = str(control).split('_')[2][:-4]
    temp = df.columns.values.tolist()
    cols = {}
    for x in temp:
        if x == "id":
            cols[x] = "AAeid"
        else:
            cols[x] = band + '_psd_' + x
    df.rename(columns=cols, inplace=True)
    df = df.reset_index(drop=True)
    return df


def get_psd():
    """
    handled psd file
    :return:
    """
    df = None
    for i in range(len(psd_control_name)):
        temp = read_psd_file(psd_control_name[i], psd_patient_name[i])
        if df is None:
            df = temp
        else:
            df = pd.merge(df, temp, on='AAeid')
    return df


def start():
    """
    main function in this script to get an whole file which get all features
    :return:
    """
    df = None
    for i in range(len(control_file)):
        temp = concact_file(control_file[i], patient_file[i])
        if df is None:
            df = temp
        else:
            del df['type']
            df = pd.merge(df, temp, on='AAeid')
    psd_df = get_psd()
    print(psd_df)
    print(df)

    df = pd.merge(psd_df, df, on='AAeid')
    df.to_csv(os.path.join(CONST.all_features_path, 'all_in_one_data.csv'), index=False)

