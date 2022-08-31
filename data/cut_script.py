import os

from tqdm import tqdm

import source.util as util
import pathlib
import soundfile as sf

IN_PATH = pathlib.Path("/home/alexandre/dataset/IDMT_FULL")
OUT_PATH = pathlib.Path("/home/alexandre/dataset/IDMT_FULL_CUT_22050")

if not OUT_PATH.exists():
    os.mkdir(OUT_PATH)

for file in tqdm(IN_PATH.rglob("*")):
    file = pathlib.Path(file)
    if file.suffix == '.wav':
        audio, rate = sf.read(file)
        cut_audio = util.cut2onset(audio, rate)
        cut_audio = cut_audio[::2]
        sf.write(OUT_PATH / file.name, cut_audio, rate // 2)
