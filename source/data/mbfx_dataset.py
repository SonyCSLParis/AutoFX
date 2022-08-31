"""
Dataset class for spectrogram from audio.
"""
import pathlib

import torch
import torchaudio
import pandas as pd
from torch import nn
from torch.utils.data import Dataset


class MBFXDataset(Dataset):
    def __init__(self, params_file: str or pathlib.Path,
                 cln_snd_dir: str or pathlib.Path, prc_snd_dir: str or pathlib.Path, rate: int,
                 resampling_rate: int = 22050, **kwargs):
        self.snd_labels = pd.read_csv(params_file)
        self.cln_snd_dir = pathlib.Path(cln_snd_dir)
        self.prc_snd_dir = pathlib.Path(prc_snd_dir)
        self.transform = nn.Sequential(
            torchaudio.transforms.Resample(rate, resampling_rate)
        )

    def __len__(self):
        return len(self.snd_labels)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        cln_snd_path = self.cln_snd_dir / (self.snd_labels.iloc[item, 0].split('_')[0] + '.wav')
        prc_snd_path = self.prc_snd_dir / (self.snd_labels.iloc[item, 0] + '.wav')
        cln_sound, rate = torchaudio.load(cln_snd_path, normalize=True)
        prc_sound, rate = torchaudio.load(prc_snd_path, normalize=True)
        params = self.snd_labels.iloc[item, 1:]
        params = torch.Tensor(params)
        cln_resampled = self.transform(cln_sound)
        prc_resampled = self.transform(prc_sound)
        return cln_resampled, prc_resampled, params
