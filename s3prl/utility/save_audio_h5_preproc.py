import os
from os.path import basename, splitext, join as path_join
import sys
import re
import json
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

# Useful for getting some stuff like Original Sampling Rate, etc. that we expect the same for ALL wavs
# Code taken from dataset.py
DUMMY_WAV_PATH = 'Session1/sentences/wav/Ses01M_script01_2/Ses01M_script01_2_F000.wav'

OUTPUT_FILE_TEMPLATE = '{}_featurized' # E.g. Session1_featurized

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


def generate_h5(sesh_folder_path, metadata_path, output_filepath):
    """Given the Session paths, load WAV one at a time batch by batch, apply torch upstream to it, and generate 
    a single H5 file"""

    # read session path and obtain WAVs list from 'meta_data'   
    with open(metadata_path, 'r') as f:       
        session_data = json.load(f)
        labels = session_data['labels']
        metadata = session_data['meta_data']

    wavs = []

    for item in metadata:
        # each item has 3 keys
        # (path: relative path to wav file from IEMOCAP_HOME; label: one of 4 'labels'; speaker: descriptive label of the specific speaker)
        wav_path = IEMOCAP_HOME + item['path']
        # label = ; speaker = ;

        wav, _ = torchaudio.load(path_join(IEMOCAP_HOME, wav_path))
        wav = RESAMPLER(wav).squeeze(0)
        # attach it to device before appending, we might as well...
        wav = torch.FloatTensor(wav.numpy()).to(DEVICE)

        wavs.append(wav)

        # Processing top be done for a singular wav at a time
        with torch.no_grad():
            feature_dict = UPSTREAM_MODEL([wav]) # Try [wav, wav, ...] N at a time until you max it out to see the biggest batch size we can tolerate
            # TODO: Save individual hidden layer data into H5 FORMAT!!!!!
            hidden_states = feature_dict["hidden_states"]            

    # wavs = [torch.randn(160000, dtype=torch.float).to(device) for _ in range(16)]
    # wavs = [torch.FloatTensor(wav).to(self.args.device) for wav in wavs]

    # PROCESS ALL WAVS AT SAME TIME? JK THAT FAILS.
    # with torch.no_grad():
    #     feature_dicts = UPSTREAM_MODEL(wavs)    # PLURAL
    #     print("computed model on all wavs.")
    #     hidden_states = feature_dict["hidden_states"]

    print("computed model on all wavs.")

    print("we looped, but please append to a file...")
    pass


def main(session_str):
    """main code"""

    sesh_folder_path = BASE_SESSION_PATH.format(session_str)
    metadata_path = SESSION_METADATA_PATH.format(session_str)
    output_filepath = sesh_folder_path + OUTPUT_FILE_TEMPLATE.format(session_str)

    # generate_h5    
    generate_h5(sesh_folder_path, metadata_path, output_filepath)    


# Note: argv[1] ie script input should be "Session1" or other valid choice in order to output the H5 file for Session1
if __name__ == "__main__":    
    main(sys.argv[1])
