# -*- coding: utf-8 -*- #
"""
    FileName     [ dataset.py ]
    Synopsis     [ the emotion classifier dataset ]
    Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""

import json
from pathlib import Path
from os.path import join as path_join
import h5py

import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import Resample

SAMPLE_RATE = 16000

# Adjust this # to effectively change how big chunks of last_layer data that we load from H5 is, and thus the frequency of file loads
LL_CHUNK_SIZE = 50

# NEW Dataset we create specifically for purposes of loading Last_layer output from existing H5 files on disk
class IemoH5LastLayerDataset(Dataset):
    def __init__(self, IEMOCAP_HOME, fold_str, train_or_test, pre_load=True):        
        # Basic Properties
        self.fold_str = fold_str
        self.session_str = fold_str.replace('fold', 'Session')
        self.type = train_or_test # Literally 'train' or 'test'
        
        # Set path to the H5 archive for the given fold        
        self.IEMOCAP_HOME = IEMOCAP_HOME
        self.h5_filename = '{}_featurized_lastlayer.h5'.format(fold_str)
        self.h5_path = path_join(IEMOCAP_HOME, self.session_str, self.h5_filename)              
        
        # Info to retrieve right away (not considered "preloading", that refers specifically to the actual layer data)
        with h5py.File(self.h5_path,"r") as hdf:
            group = hdf[train_or_test]
            data_attrs = dict(group.attrs)

            self.class_dict = json.loads(data_attrs['emotion2idx']) # deserialize the labels dict
            self.idx2emotion = {value: key for key, value in self.class_dict.items()} # create the reverse dict
            self.class_num = len(self.class_dict)
            self.num_datums = data_attrs['num_wavs']

            # TODO: CHECK IF THE BELOW EVEN WORKS OR NEED DIFF SYNTAX...
            self.ll_idxs = group['last_layer_idx'][:]            
            self.labels = group['label'][:]
            self.speakers = group['speaker'][:]

        # Initial values
        self.loaded = 0 # to start. This variable indicates num datums that have ever been loaded, even if we don't have their data in mem anymore. ASSUMPTION: we are always moving FORWARD in the dataset, never backward, which allows us to purge datums with IDX < self.loaded at any time.
        self.last_layers = [None] * self.num_datums

        # Potential preloading of all the last_layer data ... in practice this is way too big, don't do it. Rely on LL_CHUNK_SIZE behavior instead.
        self.pre_load = pre_load
        if pre_load:
            self.last_layers = self._load_all()
            self.loaded = self.num_datums # Num of layer infos that we've already loaded, IE also the idx within last_layers
        else:            
            # Load next LL_CHUNK_SIZE datums into last_layers and go on from there            
            self._chunk_lls() # internally updates self.last_layers & self.loaded

    # TODO: MAY ALSO NEED TO actively FLUSH the OLD shit from self.last_layers, WTF?!!
    # Helper method to preload the next "chunk" the data beginning at idx == self.loaded
    def _chunk_lls(self):
        # Possibly unnecessary statement, but What the hell
        if self.loaded == self.num_datums:
            return

        # SO WE KNOW WHAT TO PURGE
        orig_loaded = self.loaded

        # last_layer_list = []
        with h5py.File(self.h5_path,"r") as hdf:
            ll_raw_data = hdf[self.type]['last_layer'][:]
            # hdf['{}/{}'.format(self.type, 'last_layer')][:]
            
            end_range_loaded = min(self.loaded + LL_CHUNK_SIZE, self.num_datums)
            idx_range = range(self.loaded, end_range_loaded)
            for idx in idx_range: 
                start_idx = self.ll_idxs[idx]
                end_idx = len(ll_raw_data) if idx+1 == self.num_datums else self.ll_idxs[idx+1]
                ll = ll_raw_data[start_idx:end_idx, :]
                
                self.last_layers[idx] = ll
                # last_layer_list.append(ll)

            # Finally, update loaded count accurately
            self.loaded = end_range_loaded

        # TODO: FIX THIS LOGIC
        # TRY THIS TO GET RID OF OLD SHIT THAT IS CLOGGING MEMORY
        # if orig_loaded > 0:
        #     self.last_layers[:orig_loaded] = [None] * orig_loaded

    def _load_ll(self, idx):
        #if not hasattr(self, 'last_layers'):    # In this new chunking implementation, we always expect to have self.last_layers attr
        while (idx+1) > self.loaded:
            # IE, while the wav # that I requested is beyond what's loaded, keep chunking until we have it. NOTE if random idxs are accessed, then we're fucked.
            self._chunk_lls()
        return self.last_layers[idx]

    # TODO: Is this ever used if not preloading?
    def _load_all(self):
        '''Load the H5 last_layer data into this obj all at once...'''
        # if not hasattr(self, 'last_layers'):
        if self.loaded < self.num_datums: # In this new chunking implementation, we always expect to have self.last_layers attr
            last_layer_list = []
            with h5py.File(self.h5_path,"r") as hdf:
                ll_raw_data = hdf[self.type]['last_layer'][:]
                # hdf['{}/{}'.format(self.type, 'last_layer')][:]

                for idx in range(self.num_datums): 
                    start_idx = self.ll_idxs[idx]
                    end_idx = len(ll_raw_data) if idx+1 == self.num_datums else self.ll_idxs[idx+1]
                    ll = ll_raw_data[start_idx:end_idx, :]
                    
                    last_layer_list.append(ll)

            self.last_layers = last_layer_list
            self.loaded = self.num_datums 

        return self.last_layers

    def __getitem__(self, idx):
        idx_label = self.labels[idx]
        # emotion_label = self.idx2emotion[idx_label]
        speaker = self.speakers[idx]
        # NOTE: We wrote this to internally check if preloaded or not and do the right thing, different than how "IEMOCAPDataset" implemented it below
        ll = self._load_ll(idx)

        return ll, idx_label, speaker

    def __len__(self):
        return self.num_datums



class IEMOCAPDataset(Dataset):
    def __init__(self, data_dir, meta_path, pre_load=True):
        self.data_dir = data_dir
        self.pre_load = pre_load
        with open(meta_path, 'r') as f:
            self.data = json.load(f)
        self.class_dict = self.data['labels']
        self.idx2emotion = {value: key for key, value in self.class_dict.items()}
        self.class_num = len(self.class_dict)
        self.meta_data = self.data['meta_data']
        _, origin_sr = torchaudio.load(
            path_join(self.data_dir, self.meta_data[0]['path']))
        self.resampler = Resample(origin_sr, SAMPLE_RATE)
        if self.pre_load:
            self.wavs = self._load_all()

    def _load_wav(self, path):
        wav, _ = torchaudio.load(path_join(self.data_dir, path))
        wav = self.resampler(wav).squeeze(0)
        return wav

    def _load_all(self):
        wavforms = []
        for info in self.meta_data:
            wav = self._load_wav(info['path'])
            wavforms.append(wav)
        return wavforms

    def __getitem__(self, idx):
        label = self.meta_data[idx]['label']
        label = self.class_dict[label]
        if self.pre_load:
            wav = self.wavs[idx]
        else:
            wav = self._load_wav(self.meta_data[idx]['path'])
        return wav.numpy(), label, Path(self.meta_data[idx]['path']).stem

    def __len__(self):
        return len(self.meta_data)

def collate_fn(samples):
    return zip(*samples)
