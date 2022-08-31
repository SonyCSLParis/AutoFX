import pathlib
import pandas as pd

FX_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_modulation_delay_22050_cut")
CLEAN_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_dry_22050")


df = pd.DataFrame(columns=["clean_name"])

for file in FX_PATH.rglob("*.wav"):
    note_info = file.stem.split('-')[0:2]
    cln_file = list(CLEAN_PATH.glob(note_info[0] + '-' + note_info[1] + "-1111-*.wav"))[0]
    df.loc[file.stem] = cln_file.stem
df.to_csv(FX_PATH / 'fx2clean.csv')
