"""
Dataset class for spectrogram from audio.
"""
import pathlib

import torch
import torchaudio
import pandas as pd
from torch import nn
from torch.utils.data import Dataset

import util


class SpectroDataset(Dataset):
    def __init__(self, labels_file: str or pathlib.Path,
                 snd_dir: str or pathlib.Path, rate: int,
                 idmt: bool = False, resampling_rate: int = 16000, **kwargs):
        self.snd_labels = pd.read_csv(labels_file)
        self.snd_dir = pathlib.Path(snd_dir)
        self.idmt = idmt
        self.transform = nn.Sequential(
                            torchaudio.transforms.Resample(rate, resampling_rate),
                            torchaudio.transforms.Spectrogram(**kwargs)
                            )

    def __len__(self):
        return len(self.snd_labels)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        snd_path = self.snd_dir / self.snd_labels.iloc[item, 0]
        sound, rate = torchaudio.load(snd_path, normalize=True)
        label = self.snd_labels.iloc[item, 1]
        if self.idmt:
            sound = sound[:, int(0.45 * rate):]
            label = util.idmt_fx2class_number(label)
        spectro = self.transform(sound)
        return spectro, label
