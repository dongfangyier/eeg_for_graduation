from array import array
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tsfresh.feature_extraction import feature_calculators
from nsim import analyses1
import pyeeg

'''
命名规则
通道名+频带+特征
'''

# import matlab.engine

pick_len = 62


def eeg_get_counts(freq, length):
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
    raw = mne.io.read_raw_brainvision('/home/rbai/eegData/health_control/eyeclose/jkdz_cc_20180430_close.vhdr',
                                      preload=True)
    print()
    channel_names = []
    for i in raw.info['ch_names']:
        if i != 'Oz':
            if i != 'ECG':
                channel_names.append(i)

    bad_channels = []
    return channel_names, bad_channels


def calculate_eeg_psd_welch(raw, eid):
    psd_subfreq = {}
    raw.load_data()
    raw = raw.filter(None, 40)
    raw.resample(160, npad='auto')
    # print(raw)
    # print(raw.info)
    raw.info['bads'] = ['Oz', 'ECG']
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
    print('freq0:',freqs0)
    print('counts',counts)
    for i in range(0, pick_len):
        sum_all = sum(psd_all[i])

        rpsd_sub['delta'].append(sum(psd_all[i][0:counts[0]]) / sum_all)
        rpsd_sub['theta'].append(sum(psd_all[i][counts[0]:counts[1]]) / sum_all)
        rpsd_sub['alpha1'].append(sum(psd_all[i][counts[2]:counts[3]]) / sum_all)
        rpsd_sub['alpha2'].append(sum(psd_all[i][counts[3]:counts[4]]) / sum_all)
        rpsd_sub['beta'].append(sum(psd_all[i][counts[5]:counts[6]]) / sum_all)
        rpsd_sub['gamma'].append(sum(psd_all[i][counts[6]:]) / sum_all)

    return rpsd_sub


def hjorth(a):
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
    peak = feature_calculators.maximum(data)
    var = feature_calculators.variance(data)
    skewness = feature_calculators.skewness(data)
    kurtosis = feature_calculators.kurtosis(data)

    activity, complexity, morbidity = hjorth(data)
    return peak, var, skewness, kurtosis, activity, complexity, morbidity


def multiplication(l1, l2):
    # sum1 = 0
    # for i in range(len(a)):
    #     sum1 += a[i] * b[i]
    # return sum1
    return sum([a*b for a, b in zip(l1,l2)])


def cal_other_linear_features(raw, picks):
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


name = ['delta', 'theta', 'alpha1', 'alpha2', 'beta', 'gamma', 'full_band']

def calculate_nonlinear_features(raw, eid, columns):
    raw.load_data()
    raw = raw.filter(None, 40)
    raw.resample(160, npad='auto')

    raw.info['bads'] = ['Oz', 'ECG']
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    entity = [raw.copy().filter(0.5, 4), raw.copy().filter(4, 7), raw.copy().filter(8, 10),
              raw.copy().filter(10, 12), raw.copy().filter(13, 30), raw.copy().filter(30, 40), raw.copy()]

    res = {}
    for i in range(len(entity)):
        data = entity[i].get_data(picks=picks)
        for j in range(pick_len):
            print('change i,j:', i, j)
            #print('hurst')
            #hurst = pyeeg.hurst(list(data[j]))
            #print('fod')
            # fod = pyeeg.first_order_diff(list(data[j]))
            # pfd = pyeeg.pfd(list(data[j]), fod)
            hfd = pyeeg.hfd(list(data[j]), 10) #??????
            # hjorth_mobility, hjorth_complexity = pyeeg.hjorth(list(data[j]), fod)
            # spectral_entropy = pyeeg.spectral_entropy(X = list(data[j]), Band = [0,4,7,10,12,30,40], Fs = 160)
            # dfa = pyeeg.dfa(list(data[j]))

            #res[columns[j] + '_' + name[i] + '_' + 'hurst'] = hurst
            # res[columns[j] + '_' + name[i] + '_' + 'fod'] = fod
            # res[columns[j] + '_' + name[i] + '_' + 'pfd'] = pfd
            res[columns[j] + '_' + name[i] + '_' + 'hfd'] = hfd
            # res[columns[j] + '_' + name[i] + '_' + 'hjorth_mobility'] = hjorth_mobility
            # res[columns[j] + '_' + name[i] + '_' + 'hjorth_complexity'] = hjorth_complexity
            # res[columns[j] + '_' + name[i] + '_' + 'spectral_entropy'] = spectral_entropy
            # res[columns[j] + '_' + name[i] + '_' + 'dfa'] = dfa

    res['AAeid'] = eid
    res = dict(sorted(res.items(), key=lambda x: x[0]))

    return res

def calculate_hurst(raw, eid, columns):
    raw.load_data()
    raw = raw.filter(None, 40)
    raw.resample(160, npad='auto')

    raw.info['bads'] = ['Oz', 'ECG']
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    entity = [raw.copy().filter(0.5, 4), raw.copy().filter(4, 7), raw.copy().filter(8, 10),
              raw.copy().filter(10, 12), raw.copy().filter(13, 30), raw.copy().filter(30, 40), raw.copy()]

    res = {}
    for i in range(len(entity)):
        data = entity[i].get_data(picks=picks)
        for j in range(pick_len):
            print('change i,j:', i, j)
            print('hurst')
            hurst = pyeeg.hurst(list(data[j]))
            

            res[columns[j] + '_' + name[i] + '_' + 'hurst'] = hurst


    res['AAeid'] = eid
    res = dict(sorted(res.items(), key=lambda x: x[0]))

    return res

def calculate_linear_features(raw, eid, columns):
    raw.load_data()
    raw = raw.filter(None, 40)
    raw.resample(160, npad='auto')

    raw.info['bads'] = ['Oz', 'ECG']
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    entity = [raw.copy().filter(0.5, 4), raw.copy().filter(4, 7), raw.copy().filter(8, 10),
              raw.copy().filter(10, 12), raw.copy().filter(13, 30), raw.copy().filter(30, 40), raw.copy()]

    absolute_power, relative_power, absolute_centroid, relative_centroid = cal_other_linear_features(raw, picks)

    res = {}
    for i in range(len(entity)):
        data = entity[i].get_data(picks=picks)
        for j in range(pick_len):
            peak, var, skewness, kurtosis, activity, complexity, morbidity = cal_basic_linear_features(data[j])
            res[columns[j] + '_' + name[i] + '_' + 'peak'] = peak
            res[columns[j] + '_' + name[i] + '_' + 'var'] = var
            res[columns[j] + '_' + name[i] + '_' + 'skewness'] = skewness
            res[columns[j] + '_' + name[i] + '_' + 'kurtosis'] = kurtosis
            res[columns[j] + '_' + name[i] + '_' + 'activity'] = activity
            res[columns[j] + '_' + name[i] + '_' + 'complexity'] = complexity
            res[columns[j] + '_' + name[i] + '_' + 'morbidity'] = morbidity

            res[columns[j] + '_' + name[i] + '_' + 'absolute_power'] = absolute_power[j][i]
            res[columns[j] + '_' + name[i] + '_' + 'relative_power'] = relative_power[j][i]
            res[columns[j] + '_' + name[i] + '_' + 'absolute_centroid'] = absolute_centroid[j][i]
            res[columns[j] + '_' + name[i] + '_' + 'relative_centroid'] = relative_centroid[j][i]

    res['AAeid'] = eid
    res = dict(sorted(res.items(), key=lambda x: x[0]))

    return res


def eeg_psd(control_raw, patient_raw):
    channel_names, bad_channels = raw_data_info()

    columns = channel_names.copy()
    df = None

    counter = 0
    bigerror = []

    for (eid, raw) in control_raw.items():
        if len(raw.get_data()[0]) > 5000000:
            bigerror.append(eid)
            continue
        #res = calculate_linear_features(raw, eid, columns)
        res = calculate_nonlinear_features(raw, eid, columns)

        df1 = pd.DataFrame.from_dict(res, orient='index').T
        if counter == 0:
            df = df1
        else:
            df = pd.concat([df, df1], axis=0)
        print('control: ' + str(counter))
        counter += 1
    df.reset_index(drop=True)
    df.to_csv('data/after_c_features_nonliear_hfd.csv', index=True)

    df = None

    counter = 0
    for (eid, raw) in patient_raw.items():
        if len(raw.get_data()[0]) > 5000000:
            bigerror.append(eid)
            continue
        #res = calculate_linear_features(raw, eid, columns)
        res = calculate_nonlinear_features(raw, eid, columns)

        df1 = pd.DataFrame.from_dict(res, orient='index').T
        if counter == 0:
            df = df1
        else:
            df = pd.concat([df, df1], axis=0)
        print('patient: ' + str(counter))
        counter += 1
    df.reset_index(drop=True)
    df.to_csv('data/after_p_features_nonliear_hfd.csv', index=True)
    print(bigerror)
