import shutil

from sklearn.model_selection import train_test_split
import numpy as np
import readData
import mne
import csv
import pandas as pd
import os

'''
数据预处理
'''

# 从文件读取原始数据
def getData():
    control_raw, patient_raw, counter_c, counter_p, counter_cc, counter_pp, ab = readData.readData('/home/rbai/eegData')
    file_name = list(control_raw.keys())
    file_name += (list(patient_raw.keys()))
    print(file_name)
    print(len(file_name))
    print(len(set(file_name)))
    # len1 = []
    # for (_, x) in control_raw.items():
    #     ttt = len(x.get_data()[0])
    #     print(ttt)
    #     len1.append(ttt)
    #
    # for (_, x) in patient_raw.items():
    #     ttt = len(x.get_data()[0])
    #     print(ttt)
    #     len1.append(ttt)
    #
    # print('mean')
    # print(np.mean(len1))
    # print('max')
    #
    # print(np.max(len1))
    # print('min')
    #
    # print(np.min(len1))

    return control_raw, patient_raw, file_name

# 坏道 && 通道名
def raw_data_info(filePath):
    print('In raw_data_info')
    raw = mne.io.read_raw_brainvision(filePath + '/health_control/eyeclose/jkdz_cc_20180430_close.vhdr',
                                      preload=True)
    channel_names = []
    for i in raw.info['ch_names']:
        if i != 'Oz':
            if i != 'ECG':
                channel_names.append(i)

    bad_channels = ['Oz', 'ECG']
    print('Finish raw_data_info')
    return channel_names, bad_channels

# 存数据
def write_file(data, setName, channel_names, type):
    df = pd.DataFrame(np.array(data).T, columns=channel_names)
    if type == 'p':
        df['y'] = len(df) * [1]
    elif type == 'c':
        df['y'] = len(df) * [0]
    names=list(channel_names.copy())
    names.append('y')
    print(names)
    for channel in list(names):
        f = open('data/' + setName + '/' + channel + '.csv', 'a', newline='')
        writer = csv.writer(f)
        p = 0

        if channel == 'y':
            while (p + 256) <= len(df):
                p = p + 128
                writer.writerow(str(list(df[channel])[0]))
        else:
            while (p + 256) <= len(df):
                names = list(df[channel])[p: p + 128]
                p = p + 128
                writer.writerow(names)


writefilepath = 'data/'


# 区分训练集测试集
# 滤波 降采样后 划分 训练集测试集 并存储
def handle_data():
    if os.listdir(writefilepath).count('test') > 0:
        shutil.rmtree(writefilepath + 'test')
        os.mkdir(writefilepath + 'test')
    if os.listdir(writefilepath).count('train') > 0:
        shutil.rmtree(writefilepath + 'train')
        os.mkdir(writefilepath + 'train')

    channel_names, bad_channels = raw_data_info('/home/rbai/eegData')
    control_raw, patient_raw, filenames = getData()

    y = len(filenames) * [0]
    train, test, y1, y2 = train_test_split(filenames, y, test_size=0.5, random_state=42)
    count = 0
    baddata=0
    for name in train:
        print(count)
        count += 1
        if control_raw.keys().__contains__(name):
            raw = control_raw[name]
            raw=raw.load_data()
            if len(raw)>4000000:
                baddata+=1
                continue

            raw = raw.filter(None, 60)
            print('filter success')
            raw.resample(256, npad='auto')
            print('resample success')
            raw.info['bads'] = bad_channels
            picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
            data = raw.get_data(picks)
            write_file(data, 'train', channel_names, 'c')
        elif patient_raw.keys().__contains__(name):
            raw = patient_raw[name]
            raw=raw.load_data()
            if len(raw)>4000000:
                baddata += 1
                continue
            raw = raw.filter(None, 60)
            print('filter success')
            raw.resample(256, npad='auto')
            print('resample success')
            raw.info['bads'] = bad_channels
            picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
            data = raw.get_data(picks)
            write_file(data, 'train', channel_names, 'p')

    for name in test:
        print(count)
        count += 1
        if control_raw.keys().__contains__(name):
            raw = control_raw[name]
            raw=raw.load_data()
            if len(raw)>4000000:
                baddata += 1
                continue
            raw = raw.filter(None, 60)
            print('filter success')
            raw.resample(256, npad='auto')
            print('resample success')
            raw.info['bads'] = bad_channels
            picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
            data = raw.get_data(picks)
            write_file(data, 'test', channel_names, 'c')

        elif patient_raw.keys().__contains__(name):
            raw = patient_raw[name]
            raw=raw.load_data()
            if len(raw)>4000000:
                baddata += 1
                continue
            raw = raw.filter(None, 60)
            print('filter success')
            raw.resample(256, npad='auto')
            print('resample success')

            raw.info['bads'] = bad_channels
            picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
            data = raw.get_data(picks)
            write_file(data, 'test', channel_names, 'p')
    print(str(baddata))


handle_data()
