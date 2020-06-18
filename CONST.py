"""
define several static config in this script
"""

import os

# root path
curPath = os.path.abspath(os.path.dirname(__file__))

# eeg file path
eeg_file_path = "D:/eegData"

# the number of channels
channel_count = 62

# a sample eeg file,we can read some basic information depend on it
sample_eeg_file = os.path.join(eeg_file_path, "health_control/eyeclose/jkdz_cc_20180430_close.vhdr")

# name of bands we calculate
band_name = ['delta', 'theta', 'alpha1', 'alpha2', 'beta', 'gamma', 'full_band']

# we collect eeg about 3 minutes,if there are more than 5000000 points we think it is a bad file
bad_signal_length = 5000000

# bad channel name
bad_channel_name = ['Oz', 'ECG']

# low filter
low_filter_n = 40

# resample
resample_n = 160

# the path of calculate features saving csv
features_path = os.path.join(curPath, "cal_feature_data")

# the path for analyzing all features
all_features_path = os.path.join(curPath, "analyze_features")

# the path of select features path
select_features_path = os.path.join(all_features_path, "select_features")

# the path of classify result
classify_path = os.path.join(all_features_path, "classify")

# the path of ml model
classify_model_path = os.path.join(all_features_path, "model")

# the path of png
classify_image_path = os.path.join(all_features_path, "image")

# the path of mrmr
classify_mrmr_path = os.path.join(all_features_path, "mrmr")

# the path of classify detail result
classify_detail_path = os.path.join(classify_path, "info")





