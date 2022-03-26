import os
from os.path import basename, splitext, join as path_join
import sys
import re
import json
import h5py
import numpy as np
from librosa.util import find_files

import torch
import torchaudio
from torchaudio.transforms import Resample

import s3prl.hub as hub


# ===== Use ABSOLUTE Paths HERE FOR NOW =====

# e.g. datasets/IEMOCAP/IEMOCAP_full_release/meta_data/Session1/test_meta_data.json
# NOTE: due to IEMOCAP_preprocess.py "excited" was replaced with "happy" in labels - also, AFAIK there was only 1 label per-wav?
# NOTE: if we want, we can modify this script in order to output only the scripted or only the impromptu; only Male / Female; etc
# IE, we should view this script as a way to generate ANY useful H5 dataset we want and pass through ANY upstream we want, e.g. wav2vec2, TRILL, whatever.
# TODO: How to organize the H5?

# Place absolute path to IEMOCAP here
IEMOCAP_HOME = '/home/xadxad/work/audioproj/datasets/IEMOCAP/IEMOCAP_full_release/'

BASE_SESSION_PATH = path_join(IEMOCAP_HOME, '{}/')

SESSION_METADATA_PATH = path_join(IEMOCAP_HOME, 'meta_data/{}/test_meta_data.json')

FOLD_METADATA_PATH_TRAIN = SESSION_METADATA_PATH.replace('test_meta_data', 'train_meta_data')
FOLD_METADATA_PATH_TEST = SESSION_METADATA_PATH

# Useful for getting some stuff like Original Sampling Rate, etc. that we expect the same for ALL wavs
# Code taken from dataset.py
DUMMY_WAV_PATH = 'Session1/sentences/wav/Ses01M_script01_2/Ses01M_script01_2_F000.wav'

OUTPUT_FILE_TEMPLATE = '{}_featurized_lastlayer.h5' # E.g. fold1_featurized_lastlayer, Session1_featurized, whatev you want.

# ==========================


# Audio Processing
SAMPLE_RATE = 16000

_, ORIGIN_SR = torchaudio.load(
    path_join(IEMOCAP_HOME, DUMMY_WAV_PATH)
)

RESAMPLER = Resample(ORIGIN_SR, SAMPLE_RATE)

# Device for current computing setup
DEVICE = 'cpu'
# DEVICE = 'cuda'  # or cpu


# For reference on h5, consulted https://nicolasshu.com/appending_to_dataset_h5py.html
def load_h5(
    # session_folder_path, session_metadata_path, 
    # fold_train_metadata_path, fold_test_metadata_path,
    h5_filepath
):
    """Given the specified HDF5 pathway, load it and read its data into memory"""

    # Grab data from train + test groups
    with h5py.File(h5_filepath,"r") as hdf:
        # Groups
        train_group = hdf["train"]
        test_group = hdf["test"]

        # Attributes        
        train_attrs = dict(train_group.attrs)
        test_attrs = dict(test_group.attrs)

        # Datasets
        train_last_layers = hdf["train/last_layer"][:]
        train_ll_idx = hdf["train/last_layer_idx"][:]        
        train_labels = hdf["train/label"][:]        
        train_speakers = hdf["train/speaker"][:]
        
        test_last_layers = hdf["test/last_layer"][:]
        test_ll_idx = hdf["test/last_layer_idx"][:]
        test_labels = hdf["test/label"][:]        
        test_speakers = hdf["test/speaker"][:]

        # Example: retrieve the 0th training wav, for instance
        num_wavs = train_attrs['num_wavs']
        wav_id = 0
        start_idx = train_ll_idx[wav_id]
        end_idx = len(train_last_layers) if wav_id+1 == num_wavs else train_ll_idx[wav_id+1]
        wav_ll_data = train_last_layers[start_idx:end_idx, :]


    # NOTE: THE BELOW DOES NOT WORK - "invalid location identifier"
    # bleh1 = train_group["label"]
    # bleh2 = test_group["label"]

    # Done
    pass



def main(session_str):
    """main code"""

    session_folder_path = BASE_SESSION_PATH.format(session_str)
    session_metadata_path = SESSION_METADATA_PATH.format(session_str)

    fold_train_metadata_path = FOLD_METADATA_PATH_TRAIN.format(session_str)
    fold_test_metadata_path = FOLD_METADATA_PATH_TEST.format(session_str)

    # for h5
    fold_str = session_str.replace('Session', 'fold')
    h5_filepath = session_folder_path + OUTPUT_FILE_TEMPLATE.format(fold_str)

    # load_h5    
    load_h5(
        # session_folder_path, session_metadata_path, 
        # fold_train_metadata_path, fold_test_metadata_path,
        h5_filepath
    )    


# N.B. - Session1 corresponds to fold1 and thus also corresponds to WHICH output files are being loaded

# Note: argv[1] ie script input should be "Session1" or other valid choice in order to load the H5 file for Session1
if __name__ == "__main__":    
    main(sys.argv[1])
