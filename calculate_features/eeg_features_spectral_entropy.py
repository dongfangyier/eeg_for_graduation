import mne
import pandas as pd
import pyeeg
import CONST

"""
naming rules
channel name + band name + feature name

calculate spectral entropy
"""


pick_len = CONST.channel_count
band_name = CONST.band_name

def raw_data_info():
    """
    get channel names
    :return:
    """
    raw = mne.io.read_raw_brainvision(CONST.sample_eeg_file,
                                      preload=True)
    print()
    channel_names = []
    for i in raw.info['ch_names']:
        if i != 'Oz':
            if i != 'ECG':
                channel_names.append(i)

    return channel_names



def calculate_nonlinear_features(raw, eid, columns):
    """
    calculate entity's spectral entropy
    :param raw:
    :param eid:
    :param columns:
    :return:
    """
    raw.load_data()
    raw = raw.filter(None, CONST.low_filter_n)
    raw.resample(CONST.resample_n, npad='auto')

    raw.info['bads'] = ['Oz', 'ECG']
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    entity = [raw.copy().filter(0.5, 4), raw.copy().filter(4, 7), raw.copy().filter(8, 10),
              raw.copy().filter(10, 12), raw.copy().filter(13, 30), raw.copy().filter(30, 40), raw.copy()]

    res = {}
    for i in range(len(entity)):
        data = entity[i].get_data(picks=picks)
        for j in range(pick_len):
            print('change i,j:', i, j)
            spectral_entropy = pyeeg.spectral_entropy(X = list(data[j]), Band = [0,4,7,10,12,30,40], Fs = CONST.resample_n)

            res[columns[j] + '_' + band_name[i] + '_' + 'spectral_entropy'] = spectral_entropy

    res['AAeid'] = eid
    res = dict(sorted(res.items(), key=lambda x: x[0]))

    return res


def eeg_spectral_entropy(control_raw, patient_raw):
    """
    main function in this script to calculate spectral entropy
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
        res = calculate_nonlinear_features(raw, eid, columns)

        df1 = pd.DataFrame.from_dict(res, orient='index').T
        if counter == 0:
            df = df1
        else:
            df = pd.concat([df, df1], axis=0)
        print('control: ' + str(counter))
        counter += 1
    df.reset_index(drop=True)
    df.to_csv('data/c_features_nonliear_spectral_entropy.csv', index=True)

    df = None

    counter = 0
    for (eid, raw) in patient_raw.items():
        if len(raw.get_data()[0]) >  CONST.bad_signal_length:
            bigerror.append(eid)
            continue
        res = calculate_nonlinear_features(raw, eid, columns)

        df1 = pd.DataFrame.from_dict(res, orient='index').T
        if counter == 0:
            df = df1
        else:
            df = pd.concat([df, df1], axis=0)
        print('patient: ' + str(counter))
        counter += 1
    df.reset_index(drop=True)
    df.to_csv('data/p_features_nonliear_spectral_entropy.csv', index=True)
    print(bigerror)
