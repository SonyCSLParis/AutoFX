import pandas as pd

CONDITIONING2FX = {0: 'dry', 0.1: 'delay', 0.2: 'delay', 0.3: 'reverb',
                   0.4: 'modulation', 0.5: 'modulation', 0.6: 'modulation',
                   0.7: 'tremolo', 0.8: 'modulation', 0.9: 'distortion', 1: 'distortion'}

FX_INDEX = {'modulation': 0, 'delay': 1, 'distortion': 2, 'reverb': 3, 'tremolo': 4, 'dry': 5}

DATA_PATH = "/home/alexandre/dataset/guitar_mono_modulation_delay_22050_cut/data.csv"

df = pd.read_csv(DATA_PATH, index_col=0)
df['fx_class'] = [CONDITIONING2FX[v] for v in df['fx_class']]
df['fx_class'] = [FX_INDEX[v] for v in df['fx_class']]
df.to_csv("/home/alexandre/dataset/guitar_mono_modulation_delay_22050_cut/data_new.csv")

