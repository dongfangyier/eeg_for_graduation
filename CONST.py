"""
define several static config in this script
"""


# the number of channels
channel_count = 62

# a sample eeg file,we can read some basic information depend on it
sample_eeg_file = "/home/rbai/eegData/health_control/eyeclose/jkdz_cc_20180430_close.vhdr"

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
features_path = "data"





