import pathlib
import warnings

import numpy as np
import torch
import torchaudio

from multiband_fx import MultiBandFX
import util
import random
import pedalboard as pdb
import pandas as pd
import soundfile as sf
from tqdm.auto import tqdm

# warnings.filterwarnings("ignore", category=FutureWarning)

DATA_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_dry")
DELAY_MIN = 0
DELAY_MAX = 30
DEPTH_MIN = 0
DEPTH_MAX = 10
REGEN_MIN = -95
REGEN_MAX = 95
WIDTH_MIN = 0
WIDTH_MAX = 100
SPEED_MIN = 0.1
SPEED_MAX = 10
OUT_PATH = pathlib.Path("/home/alexandre/dataset/torchaudio_flanger")

NUM_RUNS = 20

# TODO: Could automatically adapt to new FX

dataframe = pd.DataFrame(columns=["delay", "depth", "regen", "width", "speed"], dtype='float64')
if (OUT_PATH / "params.csv").exists():
    raise ValueError("Output directory already has a params.csv file. Aborting.")
for i in tqdm(range(NUM_RUNS), position=1):
    files = []
    params_tensor = []
    for file in tqdm(DATA_PATH.iterdir(), total=1872, position=0, leave=True):
        files.append(file)
        if len(files) == 32:
            delay = torch.rand(32)
            depth = torch.rand(32)
            regen = torch.rand(32)
            width = torch.rand(32)
            speed = torch.rand(32)
            phase = torch.zeros(32)
            params_tensor.append([delay, depth, regen, width, speed])
            delay = delay * (DELAY_MAX - DELAY_MIN) + DELAY_MIN
            depth = depth * (DEPTH_MAX - DEPTH_MIN) + DEPTH_MIN
            regen = regen * (REGEN_MAX - REGEN_MIN) + REGEN_MIN
            width = width * (WIDTH_MAX - WIDTH_MIN) + WIDTH_MIN
            speed = speed * (SPEED_MAX - SPEED_MIN) + SPEED_MIN
            audio, rate = torchaudio.load(files)
            audio = torchaudio.functional.flanger(audio, rate,
                                                  delay=delay, depth=depth,
                                                  regen=regen, width=width,
                                                  speed=speed, phase=phase)
            for (j, f) in enumerate(files):
                dataframe.loc[f.stem + '_' + str(i)] = params_tensor[j]
                snd = audio[j, ::2]
                sf.write(OUT_PATH / (f.stem + '_' + str(i) + f.suffix), snd, int(rate//2))
            files = []
    dataframe.to_csv(OUT_PATH / "params.csv")
    dataframe.to_pickle(OUT_PATH / "params.pkl")



