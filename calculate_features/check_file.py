import os.path as op
import re
from io import StringIO
import configparser


def get_vhdr_info(vhdr_fname):
    """Extract all the information from the header file.
    Parameters
    ----------
    vhdr_fname : str
        Raw EEG header to be read.
    eog : list of str
        Names of channels that should be designated EOG channels. Names should
        correspond to the vhdr file.
    misc : list or tuple of str | 'auto'
        Names of channels or list of indices that should be designated
        MISC channels. Values should correspond to the electrodes
        in the vhdr file. If 'auto', units in vhdr file are used for inferring
        misc channels. Default is ``'auto'``.
    scale : float
        The scaling factor for EEG data. Unless specified otherwise by
        header file, units are in microvolts. Default scale factor is 1.
    montage : str | None | instance of Montage
        Path or instance of montage containing electrode positions. If None,
        read sensor locations from header file if present, otherwise (0, 0, 0).
        See the documentation of :func:`mne.channels.read_montage` for more
        information.
    Returns
    -------
    info : Info
        The measurement info.
    fmt : str
        The data format in the file.
    edf_info : dict
        A dict containing Brain Vision specific parameters.
    events : array, shape (n_events, 3)
        Events from the corresponding vmrk file.
    """
    ext = op.splitext(vhdr_fname)[-1]
    if ext != '.vhdr':
        raise IOError("The header file must be given to read the data, "
                      "not a file with extension '%s'." % ext)
    with open(vhdr_fname, 'rb') as f:
        header = f.readline()
        codepage = 'utf-8'

        header = header.decode('ascii', 'ignore').strip()

        settings = f.read()
        try:

            cp_setting = re.search('Codepage=(.+)',
                                   settings.decode('ascii', 'ignore'),
                                   re.IGNORECASE & re.MULTILINE)
            if cp_setting:
                codepage = cp_setting.group(1).strip()
            # BrainAmp Recorder also uses ANSI codepage
            # an ANSI codepage raises a LookupError exception
            # python recognize ANSI decoding as cp1252
            if codepage == 'ANSI':
                codepage = 'cp1252'
            settings = settings.decode(codepage)
        except UnicodeDecodeError:
            # if UTF-8 (new standard) or explicit codepage setting fails,
            # fallback to Latin-1, which is Windows default and implicit
            # standard in older recordings
            settings = settings.decode('latin-1')

    if settings.find('[Comment]') != -1:
        params, settings = settings.split('[Comment]')
    else:
        params, settings = settings, ''
    cfg = configparser.ConfigParser()
    if hasattr(cfg, 'read_file'):  # newer API
        cfg.read_file(StringIO(params))
    else:
        cfg.readfp(StringIO(params))

    # get sampling info
    # Sampling interval is given in microsec
    print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
    print([option for option in cfg['Common Infos']])
    print(cfg.get('Common Infos', 'MarkerFile'))
    print(cfg.get('Common Infos', 'DataFile'))
    print(cfg.get('Common Infos', 'DataFormat'))
    print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')

    marker = cfg.get('Common Infos', 'MarkerFile')
    eeg = cfg.get('Common Infos', 'DataFile')
    return marker[:-5], eeg[:-4]
