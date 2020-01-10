import mne
import numpy as np
import pandas as pd
import CONST
import os


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


def handle_entity(raw, eid, columns):
    """
    preprocess data
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
    data = raw.get_data(picks=picks)
    return np.array(data)


def start(control_raw, patient_raw):
    """
    main function in this script to calculate pfd
    :param control_raw:
    :param patient_raw:
    :return:
    """
    channel_names = raw_data_info()
    columns = channel_names.copy()
    bigerror = []
    cols = channel_names.copy()
    cols.append("type")
    _y = []
    _x = []

    for (eid, raw) in control_raw.items():
        if len(raw.get_data()[0]) > CONST.bad_signal_length:
            bigerror.append(eid)
            continue
        _list = handle_entity(raw, eid, columns)
        _y.append("0")
        _x.append(_list)

    counter = 0
    for (eid, raw) in patient_raw.items():
        if len(raw.get_data()[0]) > CONST.bad_signal_length:
            bigerror.append(eid)
            continue
        _list = handle_entity(raw, eid, columns)
        _y.append("1")
        _x.append(_list)
        print('patient: ' + str(counter))
        counter += 1

    print(bigerror)

    return _x, _y
