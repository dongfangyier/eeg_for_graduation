# This program is where the main function at.
# The program calls preprocessing functions, calculate psds for each
# person and use anova test to find the significant channels and sub-frequencies
import os
import sys

import mne
import numpy as np
from matplotlib import pyplot as plt

import check_file
import eeg_features_hfd

def troublesome_data(filePath):
    control_q = []
    patient_q = []
    for dirpath, _, files in os.walk(filePath):
        if 'eyeclose' in dirpath and 'health_control' in dirpath:
            #health control group
            for fname in files:
                if '.vhdr' in fname:
                    id_control = fname[:-5]
                    vmrkf, eegf = check_file.get_vhdr_info(dirpath + '/' + fname)
                    if vmrkf == eegf and vmrkf == id_control:
                        print('OK')
                    else:
                        # print('control: vhdr:' + id_control + ' vmrk: ' + vmrkf + ' eeg:' + eegf)
                        control_q.append(id_control)

        elif 'eyeclose' in dirpath and 'mdd_patient' in dirpath:
            #mdd group
            for fname in files:
                if '.vhdr' in fname:
                    id_patient = fname[:-5]
                    vmrkf, eegf = check_file.get_vhdr_info(dirpath + '/' + fname)
                    if vmrkf == eegf and vmrkf == id_patient:
                        print('OK')
                    else:
                        # print('patient: vhdr:' + id_patient + ' vmrk: ' + vmrkf + ' eeg:' + eegf)
                        patient_q.append(id_patient)

    return control_q, patient_q

def readData(filePath):
    # q contains troublesome eeg files. skip them for now
    control_q, patient_q = troublesome_data(filePath)
    #q = ['njh_after_pjk_20180725_close.vhdr', 'ccs_yb_20180813_close.vhdr', 'njh_before_pjk_20180613_close.vhdr', 'ccs_before_wjy_20180817_close.vhdr', 'ccs_after_csx_20180511_close.vhdr']
    print(patient_q)
    print('---------===========-----------')
    control_raw = {}
    patient_raw = {}
    counter_cc = 0
    counter_pp = 0
    counter_c = 0
    counter_p = 0
    ab = 0

    for dirpath, _, files in os.walk(filePath):

        if 'eyeclose' in dirpath and 'health_control' in dirpath and 'new_data' not in dirpath:
            #health control group
            print('dirpath1', dirpath)
            for fname in files:
                if '.vhdr' in fname and fname not in control_q:
                    id_control = fname[:-5]
                    counter_cc += 1
                    raw = mne.io.read_raw_brainvision(dirpath + '/' + fname, preload=False)
                    if len(raw.info['ch_names']) == 64:
                        raw.set_montage(mne.channels.read_montage("standard_1020"))
                        
                        control_raw[id_control] = raw
                    else:
                        ab += 1
                        print("Abnormal data with " +
                              str(len(raw.info['ch_names'])) + " channels. id=" + id_control)
                    print('===',counter_cc, len(control_raw))

        elif 'eyeclose' in dirpath and 'mdd_patient' in dirpath and 'new_data' not in dirpath:
            #mdd group
            print('dirpath2', dirpath)
            for fname in files:
                if '.vhdr' in fname and fname[:-5] not in patient_q:
                    id_patient = fname[:-5]
                    counter_pp += 1
                    raw = mne.io.read_raw_brainvision(dirpath + '/' + fname, preload=False)

                    if len(raw.info['ch_names']) == 64:
                        raw.set_montage(mne.channels.read_montage("standard_1020"))
                        patient_raw[id_patient] = raw
                    else:
                        ab += 1
                        print("Abnormal data with " +
                              str(len(raw.info['ch_names'])) + " channels. id=" + id_patient)

        elif 'new_data' in dirpath and 'eyeclose' in dirpath and 'health_control' in dirpath:
            #health control group new_data
            print('dirpath3', dirpath)
            
            for fname in files:
                
                if '.vhdr' in fname and fname not in control_q:
                    id_control = fname[:-5]
                    counter_c += 1
                    raw = mne.io.read_raw_brainvision(dirpath + '/' + fname, preload=False)
                    print('c!!!!!!!!!!!!!!!', len(raw.info['ch_names']))
                    if len(raw.info['ch_names']) == 64:
                        raw.set_montage(mne.channels.read_montage("standard_1020"))
                        if id_control in control_raw:
                            print('!!!!!!!!repeated')
                            ab += 1
                        control_raw[id_control] = raw
                    else:
                        ab += 1
                        print("Abnormal data with " +
                              str(len(raw.info['ch_names'])) + " channels. id=" + id_control)
                    print('===',counter_c, len(control_raw))

        elif 'new_data' in dirpath and 'eyeclose' in dirpath and 'mdd_patient' in dirpath:
            #mdd group new_data
            
            for fname in files:
                
                if '.vhdr' in fname and fname[:-5] not in patient_q:
                    id_patient = fname[:-5]
                    counter_p += 1
                    raw = mne.io.read_raw_brainvision(dirpath + '/' + fname, preload=False)
                    print('p!!!!!!!!!!!!!!!', raw.info['ch_names'])
                    if len(raw.info['ch_names']) == 64:
                        raw.set_montage(mne.channels.read_montage("standard_1020"))
                        patient_raw[id_patient] = raw
                    else:
                        ab += 1
                        print("Abnormal data with " +
                              str(len(raw.info['ch_names'])) + " channels. id=" + id_patient)

    return control_raw, patient_raw, counter_c, counter_p, counter_cc, counter_pp, ab
    #return control_q, patient_q
#raw = mne.io.read_raw_brainvision('/home/caeit/Documents/work/eeg/eegData/mdd_patient/eyeopen/njh_after_pjk_20180725_open.vhdr',preload=True)
#raw = mne.io.read_raw_brainvision('/home/caeit/Documents/work/eeg/eegData/health_control/eyeclose/jkdz_cc_20180430_close.vhdr',preload=True)

#control_raw, patient_raw = readData('/home/caeit/Documents/work/eeg/eegData')
#control_q, patient_q = readData('/home/caeit/Documents/work/eeg/eegData')
control_raw, patient_raw, counter_c, counter_p, counter_cc, counter_pp, ab = readData('/home/rbai/eegData')
#control_raw, patient_raw = readData('/home/paulbai/eeg/eegData')

print(counter_cc, counter_pp, counter_c, counter_p, ab)
eeg_features_hfd.eeg_psd(control_raw, patient_raw)

# eeg_psd_anova.psd_anova()

# eeg_psd_plot.plot_psd()


#print(control_raw)
#print('=================')
#print(patient_raw)
print('control: ' + str(len(control_raw)))
print('patient: ' + str(len(patient_raw)))
