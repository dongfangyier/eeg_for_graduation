# This program is where the main function at.
# The program calls preprocessing functions, calculate features for each
import os
import mne
import check_file
import calculate_features.eeg_calculate_linear_features as linear_feayures
import calculate_features.eeg_features_psd as psd
import calculate_features.eeg_features_dfa as dfa
import calculate_features.eeg_features_hfd as hfd
import calculate_features.eeg_features_hjorth as hjorth
import calculate_features.eeg_features_spectral_entropy as sp
import calculate_features.eeg_features_pfd as pfd

"""
sometimes our thread will be interrupted, so the script preprocessing in their own step. 
"""


def troublesome_data(filePath):
    control_q = []
    patient_q = []
    for dirpath, _, files in os.walk(filePath):
        if 'eyeclose' in dirpath and 'health_control' in dirpath:
            # health control group
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
            # mdd group
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
    # q = ['njh_after_pjk_20180725_close.vhdr', 'ccs_yb_20180813_close.vhdr', 'njh_before_pjk_20180613_close.vhdr', 'ccs_before_wjy_20180817_close.vhdr', 'ccs_after_csx_20180511_close.vhdr']
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
            # health control group
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
                    print('===', counter_cc, len(control_raw))

        elif 'eyeclose' in dirpath and 'mdd_patient' in dirpath and 'new_data' not in dirpath:
            # mdd group
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
            # health control group new_data
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
                    print('===', counter_c, len(control_raw))

        elif 'new_data' in dirpath and 'eyeclose' in dirpath and 'mdd_patient' in dirpath:
            # mdd group new_data

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



if __name__ == "__main__":
    control_raw, patient_raw, counter_c, counter_p, counter_cc, counter_pp, ab = readData('/home/rbai/eegData')

    print(counter_cc, counter_pp, counter_c, counter_p, ab)
    print('control: ' + str(len(control_raw)))
    print('patient: ' + str(len(patient_raw)))

    print("calculate features ...")
    print("linear festures ...")
    linear_feayures.eeg_linear_features(control_raw.copy(), patient_raw.copy())
    print("psd ...")
    psd.eeg_psd(control_raw.copy(), patient_raw.copy())
    print("dfa ...")
    dfa.eeg_dfa(control_raw.copy(), patient_raw.copy())
    print("hfd ...")
    hfd.eeg_hfd(control_raw.copy(), patient_raw.copy())
    print("hjorth ...")
    hjorth.eeg_hjorth(control_raw.copy(), patient_raw.copy())
    print("spectral_entropy ...")
    sp.eeg_spectral_entropy(control_raw.copy(), patient_raw.copy())
    print("pfd ...")
    pfd.eeg_pfd(control_raw.copy(), patient_raw.copy())
    print("end features ...")