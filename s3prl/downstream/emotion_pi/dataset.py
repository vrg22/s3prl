# -*- coding: utf-8 -*- #
"""
    FileName     [ dataset.py ]
    Synopsis     [ the emotion classifier dataset ]
    Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""

import os
import json
import numpy as np
from pathlib import Path
from os.path import join as path_join

import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import Resample

SAMPLE_RATE = 16000


class IEMOCAPDataset(Dataset):
    def __init__(self, data_dir, data, multiplier=2, mode='train'):
        self.data_dir = data_dir
        self.mode = mode
        self.data = data
        self.class_dict = self.data['labels']
        self.idx2emotion = {value: key for key, value in self.class_dict.items()}
        self.class_num = len(self.class_dict)
        self.meta_data = self.data['meta_data']
        _, origin_sr = torchaudio.load(
            path_join(self.data_dir, self.meta_data[0]['path']))
        self.resampler = Resample(origin_sr, SAMPLE_RATE)
        self.wavs = self._load_all()
        self.filenames = np.array([Path(self.meta_data[i]['path']).stem for i in range(len(self.wavs))])
        self.labels = np.array([self.meta_data[i]['label'] for i in range(len(self.wavs))])
        if self.mode == 'train':
            self._augmented_data(multiplier)

    def _load_wav(self, path):
        wav, _ = torchaudio.load(path_join(self.data_dir, path))
        wav = self.resampler(wav).squeeze(0)
        return wav

    def _load_all(self):
        wavforms = []
        for info in self.meta_data:
            wav = self._load_wav(info['path'])
            wavforms.append(wav.numpy())
        return wavforms

    def _augmented_data(self, multiplier=3):
        extended_wavs = []
        wavs_array = np.array(self.wavs, dtype=object)
        for k, label in enumerate(self.labels):
            label_indices = np.where(self.labels == label)[0].tolist()
            label_indices.remove(k)
            selected_wavs = np.random.choice(wavs_array[label_indices], multiplier, replace=False)
            for w in selected_wavs:
                extended_wavs.append([self.wavs[k], w])
        self.wavs = extended_wavs
        self.labels = np.repeat(self.labels, multiplier)
        self.filenames = np.repeat(self.filenames, multiplier)


    def __getitem__(self, idx):
        wav = self.wavs[idx]
        if self.mode == 'train':
            return wav[0], wav[1], self.class_dict[self.labels[idx]], self.filenames[idx]
        else:
            return wav, self.class_dict[self.labels[idx]], self.filenames[idx]

    def __len__(self):
        return len(self.wavs)

def collate_fn(samples):
    return zip(*samples)


if __name__ == "__main__":
    data_dir = '/home/tqi-gcp-voice/Research/corpora/external/IEMOCAP'
    train_path = os.path.join(data_dir, 'meta_data', 'Session1', 'train_meta_data.json')
    IEMOCAPDataset(data_dir, train_path).__getitem__(3)
