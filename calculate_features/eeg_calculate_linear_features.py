import mne
import numpy as np
import pandas as pd
from tsfresh.feature_extraction import feature_calculators
import CONST
import os

"""
naming rules
channel name + band name + feature name

calculate all linear features
"""

pick_len = CONST.channel_count

band_name = CONST.band_name


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
    raw = mne.io.read_raw_brainvision(CONST.sample_eeg_file,
                                      preload=True)
    print()
    channel_names = []
    for i in raw.info['ch_names']:
        if i not in CONST.bad_channel_name:
                channel_names.append(i)

    return channel_names


def hjorth(a):
    """
    calculate hjorth's parameter, something is wrong here
    morbidity and complexity is confusion (maybe)
    :param a:
    :return:
    """
    first_deriv = np.diff(a)
    second_deriv = np.diff(a, 2)
    var_zero = np.mean(a ** 2)
    var_d1 = np.mean(first_deriv ** 2)
    var_d2 = np.mean(second_deriv ** 2)
    activity = var_zero
    morbidity = np.sqrt(var_d1 / var_zero)
    complexity = np.sqrt(var_d2 / var_d1) / morbidity
    return activity, morbidity, complexity


def cal_basic_linear_features(data):
    """
    calculate some linear features,which is easy to complete
    :param data:
    :return:
    """
    peak = feature_calculators.maximum(data)
    var = feature_calculators.variance(data)
    skewness = feature_calculators.skewness(data)
    kurtosis = feature_calculators.kurtosis(data)

    activity, complexity, morbidity = hjorth(data)
    return peak, var, skewness, kurtosis, activity, complexity, morbidity


def multiplication(l1, l2):
    """
    a function for point multiplication
    :param l1:
    :param l2:
    :return:
    """
    return sum([a*b for a, b in zip(l1,l2)])


def cal_other_linear_features(raw, picks):
    """
    calculate power and centroid
    :param raw:
    :param picks:
    :return:
    """
    psd_all, freqs0 = mne.time_frequency.psd_welch(raw, fmin=0.5, fmax=40, picks=picks, n_fft=2048, n_jobs=1)
    counts = eeg_get_counts(freqs0, len(freqs0))
    absolute_power = []
    relative_power = []

    for i in range(0, pick_len):
        sum_all = sum(psd_all[i])
        temp = [sum(psd_all[i][0:counts[0]]), sum(psd_all[i][counts[0]:counts[1]]),
                sum(psd_all[i][counts[2]:counts[3]]),
                sum(psd_all[i][counts[3]:counts[4]]), sum(psd_all[i][counts[5]:counts[6]]), sum(psd_all[i][counts[6]:]),
                sum_all]
        absolute_power.append(temp)

        temp = [sum(psd_all[i][0:counts[0]]) / sum_all, sum(psd_all[i][counts[0]:counts[1]]) / sum_all,
                sum(psd_all[i][counts[2]:counts[3]]) / sum_all, sum(psd_all[i][counts[3]:counts[4]]) / sum_all,
                sum(psd_all[i][counts[5]:counts[6]]) / sum_all, sum(psd_all[i][counts[6]:]) / sum_all, 1]
        relative_power.append(temp)
    relative_centroid = []
    absolute_centroid = []
    for i in range(0, pick_len):
        temp = [multiplication(psd_all[i][0:counts[0]] / sum(psd_all[i]), freqs0[0:counts[0]]),
                multiplication(psd_all[i][counts[0]:counts[1]] / sum(psd_all[i]), freqs0[counts[0]:counts[1]]),
                multiplication(psd_all[i][counts[2]:counts[3]] / sum(psd_all[i]), freqs0[counts[2]:counts[3]]),
                multiplication(psd_all[i][counts[3]:counts[4]] / sum(psd_all[i]), freqs0[counts[3]:counts[4]]),
                multiplication(psd_all[i][counts[5]:counts[6]] / sum(psd_all[i]), freqs0[counts[5]:counts[6]]),
                multiplication(psd_all[i][counts[6]:] / sum(psd_all[i]), freqs0[counts[6]:])]
        sum1 = sum(temp)
        temp.append(sum1)
        absolute_centroid.append(temp)

        temp = [absolute_centroid[-1][0] / sum1,
                absolute_centroid[-1][1] / sum1,
                absolute_centroid[-1][2] / sum1,
                absolute_centroid[-1][3] / sum1,
                absolute_centroid[-1][4] / sum1,
                absolute_centroid[-1][5] / sum1, 1]

        relative_centroid.append(temp)
    return absolute_power, relative_power, absolute_centroid, relative_centroid


def calculate_linear_features(raw, eid, columns):
    """
    a function for calculate linear features
    :param raw:
    :param eid:
    :param columns:
    :return:
    """
    raw.load_data()
    raw = raw.filter(None, CONST.low_filter_n)
    raw.resample(CONST.resample_n, npad='auto')

    raw.info['bads'] = CONST.bad_channel_name
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    entity = [raw.copy().filter(0.5, 4), raw.copy().filter(4, 7), raw.copy().filter(8, 10),
              raw.copy().filter(10, 12), raw.copy().filter(13, 30), raw.copy().filter(30, 40), raw.copy()]

    absolute_power, relative_power, absolute_centroid, relative_centroid = cal_other_linear_features(raw, picks)

    res = {}
    for i in range(len(entity)):
        data = entity[i].get_data(picks=picks)
        for j in range(pick_len):
            peak, var, skewness, kurtosis, activity, complexity, morbidity = cal_basic_linear_features(data[j])
            res[columns[j] + '_' + band_name[i] + '_' + 'peak'] = peak
            res[columns[j] + '_' + band_name[i] + '_' + 'var'] = var
            res[columns[j] + '_' + band_name[i] + '_' + 'skewness'] = skewness
            res[columns[j] + '_' + band_name[i] + '_' + 'kurtosis'] = kurtosis
            res[columns[j] + '_' + band_name[i] + '_' + 'activity'] = activity
            res[columns[j] + '_' + band_name[i] + '_' + 'complexity'] = complexity
            res[columns[j] + '_' + band_name[i] + '_' + 'morbidity'] = morbidity

            res[columns[j] + '_' + band_name[i] + '_' + 'absolute_power'] = absolute_power[j][i]
            res[columns[j] + '_' + band_name[i] + '_' + 'relative_power'] = relative_power[j][i]
            res[columns[j] + '_' + band_name[i] + '_' + 'absolute_centroid'] = absolute_centroid[j][i]
            res[columns[j] + '_' + band_name[i] + '_' + 'relative_centroid'] = relative_centroid[j][i]

    res['AAeid'] = eid
    res = dict(sorted(res.items(), key=lambda x: x[0]))

    return res


def eeg_linear_features(control_raw, patient_raw):
    """
    main function in this script to calculate linear features
    :param control_raw:
    :param patient_raw:
    :return:
    """
    channel_names = raw_data_info()

    columns = channel_names.copy()
    df = None

    counter = 0
    bigerror = []
    for (eid, raw) in control_raw.items():
        if len(raw.get_data()[0]) > CONST.bad_signal_length:
            bigerror.append(eid)
            continue
        res = calculate_linear_features(raw, eid, columns)
        df1 = pd.DataFrame.from_dict(res, orient='index').T
        if counter == 0:
            df = df1
        else:
            df = pd.concat([df, df1], axis=0)
        print('control: ' + str(counter))
        counter += 1
    df.reset_index(drop=True)
    df.to_csv(os.path.join(CONST.features_path, 'c_features.csv'), index=True)

    df = None

    counter = 0
    for (eid, raw) in patient_raw.items():
        if len(raw.get_data()[0]) > CONST.bad_signal_length:
            bigerror.append(eid)
            continue
        res = calculate_linear_features(raw, eid, columns)
        df1 = pd.DataFrame.from_dict(res, orient='index').T
        if counter == 0:
            df = df1
        else:
            df = pd.concat([df, df1], axis=0)
        print('patient: ' + str(counter))
        counter += 1
    df.reset_index(drop=True)
    df.to_csv(os.path.join(CONST.features_path, 'p_features.csv'), index=True)
    print(bigerror)
