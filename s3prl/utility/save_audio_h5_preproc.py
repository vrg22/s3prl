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

# Obtain Upstream Model
WAV2VEC2_MODEL = getattr(hub, 'wav2vec2')()  # build the Wav2Vec 2.0 model with pre-trained weights
UPSTREAM_MODEL = WAV2VEC2_MODEL.to(DEVICE)

# For reference on h5, consulted https://nicolasshu.com/appending_to_dataset_h5py.html
def generate_h5(
    session_folder_path, session_metadata_path, 
    fold_train_metadata_path, fold_test_metadata_path,
    output_filepath
):
    """Given the paths, load WAV one at a time batch by batch, apply torch upstream to it, and generate 
    a single H5 file"""

    # Read path and obtain training WAVs list from 'meta_data'   
    with open(fold_train_metadata_path, 'r') as f:       
        train_data = json.load(f)
        labels = train_data['labels'] # Essentially a class dict from string label to integer class label
        train_metadata = train_data['meta_data']

    # Do the same for test data
    with open(fold_test_metadata_path, 'r') as f:       
        test_data = json.load(f)        
        test_metadata = test_data['meta_data']

    # Useful dict to have
    idx2emotion = {value: key for key, value in labels.items()}


    ### H5 FILE SETUP ###

    ## Opens file for all (r/w/append); creates if did not exist
    hdf = h5py.File(output_filepath, "a")

    # play nice w/ h5 if and when we wanna store var-len strings
    # TODO: if this ain't great, see https://docs.h5py.org/en/stable/special.html
    vlen_string_dt = h5py.special_dtype(vlen=str)

    # Create macro level groups within our file. Specifically, let's create a "train" group and a "test" group for each fold/Session    
    train_group = hdf.create_group("train")
    train_group.attrs["num_wavs"] = len(train_metadata)

    test_group = hdf.create_group("test")
    test_group.attrs["num_wavs"] = len(test_metadata)    

    ################### Setup TRAIN group w/ relevant datasets
    ## [1] "last_layer" dataset. The individual WAV tensors are float of shape (?, 768), so we essentially gotta concat these together.
    # Axis 0 of the below simply stores the frames; Axis 1 is gonna be 768
    train_ll_dset = train_group.create_dataset(
        name  = "last_layer",
        shape = (0, 768), # NOTE: Before every append / write operation, MUST resize this dset as needed to be able to stack the encodings!
        maxshape = (None, 768), # None means that this dimension can be extended to basically unlimited length as needed
        dtype = "float64" # CHECK
    )

    # TODO: Instead of below workaround, investigate if there is a way to use variable length tensors later. Perhaps that's somehow speedier too? POTENTIAL ISSUE tho: H5 may not play nice and might force you to extract all at once 
    # TODO: write a helper read method later for use elsewhere
    ## [1A] "last_layer_idx" dataset which stores start idx within Axis 0 of the "last_layer" dataset of the layer encoding. Use this to effectively index into last_layer dataset to pull out encoding for arbitrary Wav #N within the set
    train_layer_idx_dset = train_group.create_dataset(
        name  = "last_layer_idx",
        shape = len(train_metadata),
        maxshape = len(train_metadata),
        dtype = "i"
    )

    ## [2] "label" dataset with corresponding emotion label (INTEGER)
    train_lbl_dset = train_group.create_dataset(
        name  = "label",
        shape = len(train_metadata),
        maxshape = len(train_metadata),
        dtype = "i"
    )

    ## [3] "speaker" dataset with corresponding speaker's name (STRING)
    train_spkr_dset = train_group.create_dataset(
        name  = "speaker",
        shape = len(train_metadata),
        maxshape = len(train_metadata),
        dtype = vlen_string_dt
    )
    
        
    ################### Setup TEST group w/ relevant datasets
    ## [1] "last_layer" dataset. The individual WAV tensors are float of shape (?, 768), so we essentially gotta concat these together.
    # Axis 0 of the below simply stores the frames; Axis 1 is gonna be 768
    test_ll_dset = test_group.create_dataset(
        name  = "last_layer",
        shape = (0, 768), # NOTE: Before every append / write operation, MUST resize this dset as needed to be able to stack the encodings!
        maxshape = (None, 768), # None means that this dimension can be extended to basically unlimited length as needed
        dtype = "float64" # CHECK
    )

    ## [1A] "last_layer_idx" dataset which stores start idx within Axis 0 of the "last_layer" dataset of the layer encoding. Use this to effectively index into last_layer dataset to pull out encoding for arbitrary Wav #N within the set
    test_layer_idx_dset = test_group.create_dataset(
        name  = "last_layer_idx",
        shape = len(test_metadata),
        maxshape = len(test_metadata),
        dtype = "i"
    )

    ## [2] "label" dataset with corresponding emotion label (INTEGER)
    test_lbl_dset = test_group.create_dataset(
        name  = "label",
        shape = len(test_metadata),
        maxshape = len(test_metadata),
        dtype = "i" # CHECK
    )

    ## [3] "speaker" dataset with corresponding speaker's name (STRING)
    test_spkr_dset = test_group.create_dataset(
        name  = "speaker",
        shape = len(test_metadata),
        maxshape = len(test_metadata),
        dtype = vlen_string_dt # CHECK
    )


    # Loop to save *TRAINING* WAVs into H5
    train_wavs = []
    start_idx = 0
    for train_wav_id, item in enumerate(train_metadata):
        # each item has 3 keys
        # (path: relative path to wav file from IEMOCAP_HOME; label: one of 4 'labels'; speaker: descriptive label of the specific speaker)
        wav_path = IEMOCAP_HOME + item['path']
        train_lbl_dset[train_wav_id] = labels[item['label']]
        train_spkr_dset[train_wav_id] = item['speaker']

        wav, _ = torchaudio.load(path_join(IEMOCAP_HOME, wav_path))
        wav = RESAMPLER(wav).squeeze(0)
        # attach it to device before appending, we might as well...
        wav = torch.FloatTensor(wav.numpy()).to(DEVICE)

        train_wavs.append(wav)

        # Processing to be done for a singular wav at a time
        with torch.no_grad():
            feature_dict = UPSTREAM_MODEL([wav]) # Try [wav, wav, ...] N at a time until you max it out to see the biggest batch size we can tolerate            
            # hidden_states = feature_dict["hidden_states"]

            # Grab only the last layer
            last_layer = feature_dict["last_hidden_state"]

            # Store start_idx for this id in the helper dataset + update start_idx
            train_layer_idx_dset[train_wav_id] = start_idx
            seq_len = last_layer.shape[1]
            start_idx += seq_len # next start idx

            # TODO: Does resize dramatically slow things down??? If so may need to preprocess up front...
            # Resize the last_layer dset to add seq_len amount of space
            train_ll_dset.resize(train_ll_dset.shape[0] + seq_len, axis=0)

            # Stack the meaningful axes of last_layer to dset
            train_ll_dset[-seq_len:] = torch.squeeze(last_layer)

        ############ FOR TESTING PURPOSES ONLY!!!!! ##############
        # if train_wav_id == 5:
        #     sys.exit()

    print("--DONE PROCESSING training WAVS--")



    # ANOTHER Loop to save *TESTING* WAVs into H5
    test_wavs = []
    start_idx = 0
    for test_wav_id, item in enumerate(test_metadata):
        # each item has 3 keys
        # (path: relative path to wav file from IEMOCAP_HOME; label: one of 4 'labels'; speaker: descriptive label of the specific speaker)
        wav_path = IEMOCAP_HOME + item['path']        
        test_lbl_dset[test_wav_id] = labels[item['label']]
        test_spkr_dset[test_wav_id] = item['speaker']

        wav, _ = torchaudio.load(path_join(IEMOCAP_HOME, wav_path))
        wav = RESAMPLER(wav).squeeze(0)
        # attach it to device before appending, we might as well...
        wav = torch.FloatTensor(wav.numpy()).to(DEVICE)

        test_wavs.append(wav)

        # Processing to be done for a singular wav at a time
        with torch.no_grad():
            feature_dict = UPSTREAM_MODEL([wav]) # Try [wav, wav, ...] N at a time until you max it out to see the biggest batch size we can tolerate            
            # hidden_states = feature_dict["hidden_states"]

            # Grab only the last layer
            last_layer = feature_dict["last_hidden_state"]

            # Store start_idx for this id in the helper dataset + update start_idx
            test_layer_idx_dset[test_wav_id] = start_idx
            seq_len = last_layer.shape[1]
            start_idx += seq_len # next start idx
            
            # Resize the last_layer dset to add seq_len amount of space
            test_ll_dset.resize(test_ll_dset.shape[0] + seq_len, axis=0)

            # Stack the meaningful axes of last_layer to dset
            test_ll_dset[-seq_len:] = torch.squeeze(last_layer)


    print("--DONE PROCESSING testing WAVS--")



    # wavs = [torch.randn(160000, dtype=torch.float).to(device) for _ in range(16)]
    # wavs = [torch.FloatTensor(wav).to(self.args.device) for wav in wavs]

    # PROCESS ALL WAVS AT SAME TIME? JK THAT FAILS.
    # with torch.no_grad():
    #     feature_dicts = UPSTREAM_MODEL(wavs)    # PLURAL
    #     print("computed model on all wavs.")
    #     hidden_states = feature_dict["hidden_states"]

    
    
    
    # print("computed model on all wavs.")
    
    print("--CLOSING H5 FILE--")

    ## Close H5 file
    hdf.close()

    # DONE
    pass


def main(session_str):
    """main code"""

    session_folder_path = BASE_SESSION_PATH.format(session_str)
    session_metadata_path = SESSION_METADATA_PATH.format(session_str)

    fold_train_metadata_path = FOLD_METADATA_PATH_TRAIN.format(session_str)
    fold_test_metadata_path = FOLD_METADATA_PATH_TEST.format(session_str)

    # for h5
    fold_str = session_str.replace('Session', 'fold')
    output_filepath = session_folder_path + OUTPUT_FILE_TEMPLATE.format(fold_str)

    # generate_h5    
    generate_h5(
        session_folder_path, session_metadata_path, 
        fold_train_metadata_path, fold_test_metadata_path,
        output_filepath
    )    


# ************ TODO: just change input args / code SLIGHTLY to allow for us to save folds + train or test hd5s instead of just Session****************
# N.B. - Session1 corresponds to fold1 and thus also corresponds to WHICH output files are being created

# Note: argv[1] ie script input should be "Session1" or other valid choice in order to output the H5 file for Session1
if __name__ == "__main__":    
    main(sys.argv[1])
