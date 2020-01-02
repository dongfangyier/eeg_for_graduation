
#this program generate eeg data cut in either 3s or 1s
import numpy as np
import mne
import check_file
import os, sys
import pandas as pd
from multiprocessing import Process
from sklearn.model_selection import train_test_split
import shutil

# import eeg_tsfresh_calcFeatures
import threading
import csv



'''
读取脑电文件，分割成3s（1s）
'''

def troublesome_data(filePath):
	print('In troublesome_data')
	control_q = []
	patient_q = []
	for dirpath, dirs, files in os.walk(filePath):

		if 'eyeclose' in dirpath and 'health_control' in dirpath:
			# health control group
			for fname in files:
				if '.vhdr' in fname:
					id_control = fname[:-5]
					vmrkf, eegf = check_file.get_vhdr_info(dirpath + '/' + fname)
					if vmrkf == eegf and vmrkf == id_control:
						print('OK')
					else:
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
						patient_q.append(id_patient)
	print('Finish troublesome_data')
	control_q.append('jkdz_cc_01_20180430_close.vhdr')
	control_q.append('jkdz_cc_02_20180430_close.vhdr')
	control_q.append('jkdz_cc_03_20180430_close.vhdr')

	return control_q, patient_q


def read_data(filePath):
	print('In read_data')
	# q contains troublesome eeg files. skip them for now
	control_q, patient_q = troublesome_data(filePath)
	# q = ['njh_after_pjk_20180725_close.vhdr', 'ccs_yb_20180813_close.vhdr', 'njh_before_pjk_20180613_close.vhdr', 'ccs_before_wjy_20180817_close.vhdr', 'ccs_after_csx_20180511_close.vhdr']
	control_raw = {}
	patient_raw = {}
	control_counter = 0
	patient_counter = 0
	for dirpath, dirs, files in os.walk(filePath):

		if 'eyeclose' in dirpath and 'health_control' in dirpath:
			# health control group
			for fname in files:


				if '.vhdr' in fname and fname not in control_q:
					control_counter += 1
					print("CONTROL: No. ", control_counter)
					id_control = fname[:-5]

					raw = mne.io.read_raw_brainvision(dirpath + '/' + fname, preload=False)
					if len(raw.info['ch_names']) == 65:
						raw.set_montage(mne.channels.read_montage("standard_1020"))
						control_raw[id_control] = raw
					else:
						print("Abnormal data with " + str(len(raw.info['ch_names'])) + " channels. id=" + id_control)

		elif 'eyeclose' in dirpath and 'mdd_patient' in dirpath:
			# mdd group
			for fname in files:


				if '.vhdr' in fname and fname[:-5] not in patient_q:
					patient_counter += 1
					print("PATIENT: No. ", patient_counter)

					id_patient = fname[:-5]

					raw = mne.io.read_raw_brainvision(dirpath + '/' + fname, preload=False)

					if len(raw.info['ch_names']) == 65:
						raw.set_montage(mne.channels.read_montage("standard_1020"))
						patient_raw[id_patient] = raw
					else:
						print("Abnormal data with " + str(len(raw.info['ch_names'])) + " channels. id=" + id_patient)
	print('Finish read_data')
	return control_raw, patient_raw

def raw_data_info(filePath):
	print('In raw_data_info')
	raw = mne.io.read_raw_brainvision(filePath + '/health_control/eyeclose/jkdz_cc_20180430_close.vhdr',
									  preload=True)
	# channel_names = raw.info['ch_names']

	channel_names = []
	for i in raw.info['ch_names']:
		if i != 'Oz':
			if i != 'ECG':
				channel_names.append(i)

	bad_channels = ['Oz', 'ECG']
	print('Finish raw_data_info')
	return channel_names, bad_channels

#this function loads data for each raw object, filter data, resample data
# to 512Hz, eliminate bad channels and return raw object and picks
def preprocess_data(raw, bad_channels):
	print('In preprocess_data')
	raw.load_data()
	raw = raw.filter(None, 60)
	print('filter success')
	raw.resample(512, npad='auto')
	# if len(raw)>100000 or len(raw)<19000:
	#	 print(counter)
	#	 fileObject = open(str(counter)+'.txt', 'w')
	#	 fileObject.write(str(counter))
	#	 fileObject.close()
	print('success resample')
	raw.info['bads'] = bad_channels
	picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
	print('Finish preprocess_data')
	return raw, picks
	# temp = len(raw)
	# temp_res = []
	# i = 0
	# while temp-(512*3) >= 0:
	#	 x = raw.get_data(picks, start=i, stop=i + (512*3))
	#	 i += (512*3)
	#	 temp -= (512*3)
	#	 x = np.array(x).T
	#	 temp_res.append(x)
	#	 print(temp)

	# # raw = raw.get_data(picks,start=9500,stop=19000)
	# # raw=np.array(raw).T

	# return temp_res

def cut_data(data_cut_len, control_raw, patient_raw, channel_names, bad_channels, writefilepath):
	print('In cut_data')
	columns = channel_names.copy()
	columns.insert(0, 'time')
	columns.insert(0, 'id')
	columns = columns[:-1]
	columns.append('y')
	data_to_save = pd.DataFrame(columns = columns)
	counter = 0
	control_counter = 0
	patient_counter = 0
	for (eid, raw) in control_raw.items():
		control_counter += 1
		print('Preprocess data', control_counter)
		filtered_raw, picks = preprocess_data(raw, bad_channels)
		temp_cut = []
		numOfCut = int(len(raw)/(512*data_cut_len))
		for n in range(numOfCut):
			cut = raw.get_data(picks, start = n * 512 * data_cut_len, stop = (n+1) * 512 * data_cut_len)
			cut = np.array(cut).T
			temp_cut.append(cut)
			f = open(writefilepath + 'data_cut/control_data_' + str(counter) + '.csv', 'w', newline='')
			writer = csv.writer(f)
			writer.writerow(columns)
			time = 0
			for c in cut:
				c = list(c)
				c.insert(0, time)
				c.insert(0, counter)
				time += 1
				c.append('1')
				writer.writerow(c)
			f.close()
			counter += 1

	counter = 0
	for (eid, raw) in patient_raw.items():
		patient_counter += 1
		print('Preprocess data', patient_counter)
		filtered_raw, picks = preprocess_data(raw, bad_channels)
		temp_cut = []
		numOfCut = int(len(raw)/(512*data_cut_len))
		for n in range(numOfCut):
			cut = raw.get_data(picks, start = n * 512 * data_cut_len, stop = (n+1) * 512 * data_cut_len)
			cut = np.array(cut).T
			temp_cut.append(cut)
			f = open(writefilepath + 'data_cut/patient_data_' + str(counter) + '.csv', 'w', newline='')
			writer = csv.writer(f)
			writer.writerow(columns)
			time = 0
			for c in cut:
				c = list(c)
				c.insert(0, time)
				c.insert(0, counter)
				time += 1
				c.append('0')
				writer.writerow(c)
			f.close()
			counter += 1
	print('Finish cut_data')

def write_file(writefilepath, setName, fname, channel_names):
	df = pd.read_csv(writefilepath + 'data_cut/' + fname)
	print(fname)

	for channel in list(df):
		f = open(writefilepath + '/' + setName + '/' + channel + '.csv', 'a', newline='')
		writer = csv.writer(f)
		p = 0

		if channel == 'y' or channel == 'id':
			while (p+128) <= len(df):
				#temp = list(df[channel])[p: p+128]
				p = p+64
				writer.writerow(str(list(df[channel])[0]))
		else:
			while (p+128) <= len(df):
				temp = list(df[channel])[p: p+128]
				p = p+64
				writer.writerow(temp)





def write_train_test_file(writefilepath, channel_names):
	print('In write_train_test_file')
	X = os.listdir(writefilepath + 'data_cut/')
	if os.listdir(writefilepath).count('test') > 0:
		shutil.rmtree(writefilepath + 'test')
		os.mkdir(writefilepath + 'test')
	if os.listdir(writefilepath).count('train') > 0:
		shutil.rmtree(writefilepath + 'train')
		os.mkdir(writefilepath + 'train')

	y = len(X)*[0]
	train, test, y1, y2= train_test_split(X,y,test_size=0.3, random_state=42)
	counter = 0

	for fname in train:
		print(counter, '/', len(train), fname)
		counter += 1
		write_file(writefilepath, 'train', fname, channel_names)
	counter = 0
	for fname in test:
		print(counter, '/', len(test), fname)
		counter += 1
		write_file(writefilepath, 'test', fname, channel_names)






def generate_data(readfilepath, data_cut_len, writefilepath):
	control_raw, patient_raw = read_data(readfilepath)
	print('~~~~~~~~~~~~~~~~~~~~~~~~~~Done1')
	channel_names, bad_channels = raw_data_info(readfilepath)
	print('~~~~~~~~~~~~~~~~~~~~~~~~~~Done2')
	cut_data(data_cut_len, control_raw, patient_raw, channel_names, bad_channels, writefilepath)
	print('~~~~~~~~~~~~~~~~~~~~~~~~~~Done3')
	write_train_test_file(writefilepath, channel_names)
	print('~~~~~~~~~~~~~~~~~~~~~~~~~~Done4')

readfilepath = '/home/rbai/eegData/'
data_cut_len = 3
writefilepath = '/home/rbai/eeg_lstm/data/' + str(data_cut_len) + 's/'
generate_data(readfilepath, data_cut_len, writefilepath)
