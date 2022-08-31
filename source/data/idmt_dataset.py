import pathlib
import torch
import torchaudio
import pandas as pd
from torch import nn
from torch.utils.data import Dataset


class IDMTDataset(Dataset):
    def __init__(self, fx2clean_file: str or pathlib.Path, path2cln: str or pathlib.Path,
                 path2fx: str or pathlib.Path):
        self.fx2clean = pd.read_csv(fx2clean_file)
        if not pathlib.Path(path2cln).is_dir():
            raise TypeError("{} is not a directory.".format(path2cln))
        self.path2cln = pathlib.Path(path2cln)
        if not pathlib.Path(path2fx).is_dir():
            raise TypeError("{} is not a directory.".format(path2fx))
        self.path2fx = pathlib.Path(path2fx)

    def __len__(self):
        return len(self.fx2clean)

    def __getitem__(self, item):
        # if torch.is_tensor(item):
        #    item = item.tolist()
        cln_snd_path = self.path2cln / (self.fx2clean.iloc[item, 1] + '.wav')
        fx_snd_path = self.path2fx / (self.fx2clean.iloc[item, 0] + '.wav')
        cln_sound, _ = torchaudio.load(cln_snd_path, normalize=True)
        fx_sound, _ = torchaudio.load(fx_snd_path, normalize=True)
        return cln_sound, fx_sound
