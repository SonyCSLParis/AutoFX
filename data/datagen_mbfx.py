import os
import pathlib
import warnings

import numpy as np
import torch

from source.multiband_fx import MultiBandFX
import source.util as util
import random
import pedalboard as pdb
import pandas as pd
import soundfile as sf
from tqdm.auto import tqdm
from source.models.custom_distortion import CustomDistortion

# warnings.filterwarnings("ignore", category=FutureWarning)

DATA_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_dry_22050_cut")
NUM_BANDS = 1
NUM_CHANGED_BANDS = 1
# FX = [pdb.Delay()]
FX = CustomDistortion()
PARAM_RANGE = [(0, 60),
               (50, 500), (-10, 10), (0.5, 2),
               (500, 2000), (-10, 10), (0.5, 2)]
PARAMS = ["p-drive_db",
          "p-cutoff_frequency_hz_lo", "p-gain_db_lo", "p-q_lo",
          "p-cutoff_frequency_hz_hi", "p-gain_db_hi", "p-q_hi"]
OUT_PATH = pathlib.Path("/home/alexandre/dataset/distortion_guitar_mono_cut2")

NUM_RUNS = 20

# TODO: Could automatically adapt to new FX

dataframe = pd.DataFrame(columns=PARAMS, dtype='float64')
# mbfx = MultiBandFX(FX, NUM_BANDS)
mbfx = FX
if not(OUT_PATH.exists()):
    os.mkdir(OUT_PATH)
if (OUT_PATH / "params.csv").exists():
    raise ValueError("Output directory already has a params.csv file. Aborting.")
for i in tqdm(range(NUM_RUNS), position=1):
    for file in tqdm(DATA_PATH.rglob('*.wav'), total=1872, position=0, leave=True):
        params = [round(random.random(), 2) for p in range(len(PARAMS))]
        # params.append(0)    #  freeze_mode for reverb
        mbfx.set_fx_params(params, param_range=PARAM_RANGE, flat=True)
        audio, rate = util.read_audio(file, normalize=True, add_noise=False)
        # processed = np.zeros((1, int(5.12*rate)))   # zero-pad to keep reverb tail (up to 5s)
        # processed[0, :len(audio[0])] = audio
        processed = mbfx.process(audio, rate)
        dataframe.loc[file.stem + '_' + str(i)] = params
        # audio = audio[0, ::2]
        processed = processed[0] / torch.max(torch.abs(processed[0]))
        sf.write(OUT_PATH / (file.stem + '_' + str(i) + file.suffix), processed, int(rate//1))      # CAREFUL WITH SUBSAMPLING
    dataframe.to_csv(OUT_PATH / "params.csv")
    dataframe.to_pickle(OUT_PATH / "params.pkl")
with open(OUT_PATH / "config.txt", 'w') as f:
    f.write(str(PARAM_RANGE))




