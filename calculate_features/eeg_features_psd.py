import mne
import numpy as np
import pandas as pd
import CONST
import os

pick_len = CONST.channel_count


def eeg_get_counts(freq, length):
    """
    get the limit of the frequency list
    :param freq: frequency list
    :param length: len(freq)
    :return:a list of limit
    """
    counts = []
    for i in range(0, length):
        if len(counts) == 0 and freq[i] >= 4:
            counts.append(i)
        if len(counts) == 1 and freq[i] >= 7:
            counts.append(i)
        if len(counts) == 2 and freq[i] >= 8:
            counts.append(i)
        if len(counts) == 3 and freq[i] >= 10:
            counts.append(i)
        if len(counts) == 4 and freq[i] >= 12:
            counts.append(i)
        if len(counts) == 5 and freq[i] >= 13:
            counts.append(i)
        if len(counts) == 6 and freq[i] >= 30:
            counts.append(i)
            break
    return counts


def raw_data_info():
    """
    get channel name
    :return:
    """
    raw = mne.io.read_raw_brainvision(CONST.sample_eeg_file, preload=True)
    print()
    channel_names = []
    for i in raw.info['ch_names']:
        if i not in CONST.bad_channel_name:
                channel_names.append(i)

    return channel_names


def calculate_eeg_psd_welch(raw, eid):
    """
    calculate psd using welch method
    :param raw:
    :param eid:
    :return:
    """
    raw.load_data()
    raw = raw.filter(None, CONST.low_filter_n)
    raw.resample(CONST.resample_n, npad='auto')

    raw.info['bads'] = CONST.bad_channel_name
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    pick_len = len(picks)
    psd_all, freqs0 = mne.time_frequency.psd_welch(raw, fmin=0.5, fmax=40, picks=picks, n_fft=2048, n_jobs=1)

    rpsd_sub = {}
    rpsd_sub['delta'] = []
    rpsd_sub['theta'] = []
    rpsd_sub['alpha1'] = []
    rpsd_sub['alpha2'] = []
    rpsd_sub['beta'] = []
    rpsd_sub['gamma'] = []

    counts = eeg_get_counts(freqs0, len(freqs0))
    print(counts)
    for i in range(0, pick_len):
        sum_all = sum(psd_all[i])

        rpsd_sub['delta'].append(sum(psd_all[i][0:counts[0]]) / sum_all)
        rpsd_sub['theta'].append(sum(psd_all[i][counts[0]:counts[1]]) / sum_all)
        rpsd_sub['alpha1'].append(sum(psd_all[i][counts[2]:counts[3]]) / sum_all)
        rpsd_sub['alpha2'].append(sum(psd_all[i][counts[3]:counts[4]]) / sum_all)
        rpsd_sub['beta'].append(sum(psd_all[i][counts[5]:counts[6]]) / sum_all)
        rpsd_sub['gamma'].append(sum(psd_all[i][counts[6]:]) / sum_all)

    return rpsd_sub


def eeg_psd(control_raw, patient_raw):
    """
    main function to calculate psd of eeg
    :param control_raw:
    :param patient_raw:
    :return:
    """
    channel_names = raw_data_info()

    columns = channel_names.copy()
    columns.insert(0, 'groupId')
    columns.insert(0, 'id')
    columns = columns[:-1]
    columns.append('average')
    df_c_delta = pd.DataFrame(columns=columns)
    df_c_theta = pd.DataFrame(columns=columns)
    df_c_alpha1 = pd.DataFrame(columns=columns)
    df_c_alpha2 = pd.DataFrame(columns=columns)
    df_c_beta = pd.DataFrame(columns=columns)
    df_c_gamma = pd.DataFrame(columns=columns)

    df_p_delta = pd.DataFrame(columns=columns)
    df_p_theta = pd.DataFrame(columns=columns)
    df_p_alpha1 = pd.DataFrame(columns=columns)
    df_p_alpha2 = pd.DataFrame(columns=columns)
    df_p_beta = pd.DataFrame(columns=columns)
    df_p_gamma = pd.DataFrame(columns=columns)
    bigerror=[]
    counter = 0
    for (eid, raw) in control_raw.items():
        if len(raw.get_data()[0]) > CONST.bad_signal_length:
            bigerror.append(eid)
            continue
        psd_subfreq = calculate_eeg_psd_welch(raw, eid)

        print('control: ' + str(counter))
        counter += 1
        temp = psd_subfreq['delta'].copy()

        temp.insert(0, 0)
        temp.insert(0, eid)
        temp.append(np.mean(psd_subfreq['delta']))

        df_c_delta.loc[len(df_c_delta)] = temp

        temp = psd_subfreq['theta'].copy()
        temp.insert(0, 0)
        temp.insert(0, eid)
        temp.append(np.mean(psd_subfreq['theta']))

        df_c_theta.loc[len(df_c_theta)] = temp

        temp = psd_subfreq['alpha1'].copy()
        temp.insert(0, 0)
        temp.insert(0, eid)
        temp.append(np.mean(psd_subfreq['alpha1']))

        df_c_alpha1.loc[len(df_c_alpha1)] = temp

        temp = psd_subfreq['alpha2'].copy()
        temp.insert(0, 0)
        temp.insert(0, eid)
        temp.append(np.mean(psd_subfreq['alpha2']))

        df_c_alpha2.loc[len(df_c_alpha2)] = temp

        temp = psd_subfreq['beta'].copy()
        temp.insert(0, 0)
        temp.insert(0, eid)
        temp.append(np.mean(psd_subfreq['beta']))

        df_c_beta.loc[len(df_c_beta)] = temp

        temp = psd_subfreq['gamma'].copy()
        temp.insert(0, 0)
        temp.insert(0, eid)
        temp.append(np.mean(psd_subfreq['gamma']))

        df_c_gamma.loc[len(df_c_gamma)] = temp

    df_c_delta.to_csv('psd_c_delta.csv', index=True)
    df_c_theta.to_csv('psd_c_theta.csv', index=True)
    df_c_alpha1.to_csv('psd_c_alpha1.csv', index=True)
    df_c_alpha2.to_csv('psd_c_alpha2.csv', index=True)
    df_c_beta.to_csv('psd_c_beta.csv', index=True)
    df_c_gamma.to_csv('psd_c_gamma.csv', index=True)

    counter = 0
    for (eid, raw) in patient_raw.items():
        if len(raw.get_data()[0]) > CONST.bad_signal_length:
            bigerror.append(eid)
            continue
        psd_subfreq = calculate_eeg_psd_welch(raw, eid)
        print('patient #' + str(counter) + ': ' + eid)
        counter += 1

        temp = psd_subfreq['delta'].copy()

        temp.insert(0, 1)
        temp.insert(0, eid)
        temp.append(np.mean(psd_subfreq['delta']))

        df_p_delta.loc[len(df_p_delta)] = temp

        temp = psd_subfreq['theta'].copy()
        temp.insert(0, 1)
        temp.insert(0, eid)
        temp.append(np.mean(psd_subfreq['theta']))

        df_p_theta.loc[len(df_p_theta)] = temp

        temp = psd_subfreq['alpha1'].copy()
        temp.insert(0, 1)
        temp.insert(0, eid)
        temp.append(np.mean(psd_subfreq['alpha1']))

        df_p_alpha1.loc[len(df_p_alpha1)] = temp

        temp = psd_subfreq['alpha2'].copy()
        temp.insert(0, 1)
        temp.insert(0, eid)
        temp.append(np.mean(psd_subfreq['alpha2']))

        df_p_alpha2.loc[len(df_p_alpha2)] = temp

        temp = psd_subfreq['beta'].copy()
        temp.insert(0, 1)
        temp.insert(0, eid)
        temp.append(np.mean(psd_subfreq['beta']))

        df_p_beta.loc[len(df_p_beta)] = temp

        temp = psd_subfreq['gamma'].copy()
        temp.insert(0, 1)
        temp.insert(0, eid)
        temp.append(np.mean(psd_subfreq['gamma']))

        df_p_gamma.loc[len(df_p_gamma)] = temp

    df_p_delta.to_csv(os.path.join(CONST.features_path, 'psd_p_delta.csv'), index=True)
    df_p_theta.to_csv(os.path.join(CONST.features_path, 'psd_p_theta.csv'), index=True)
    df_p_alpha1.to_csv(os.path.join(CONST.features_path, 'psd_p_alpha1.csv'), index=True)
    df_p_alpha2.to_csv(os.path.join(CONST.features_path, 'psd_p_alpha2.csv'), index=True)
    df_p_beta.to_csv(os.path.join(CONST.features_path, 'psd_p_beta.csv'), index=True)
    df_p_gamma.to_csv(os.path.join(CONST.features_path, 'psd_p_gamma.csv'), index=True)
    print(bigerror)
