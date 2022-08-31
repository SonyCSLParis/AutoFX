import argparse

import torch
import torch.nn.functional as F
import pandas as pd
import pathlib
import sys

import torchaudio
import tqdm

COLUMNS = ['c-dry', 'c-feedback_delay', 'c-slapback_delay',
           'c-reverb', 'c-chorus', 'c-flanger', 'c-phaser',
           'c-tremolo', 'c-vibrato', 'c-distortion', 'c-overdrive', 'fx_class']


def main(parser):
    args = vars(parser.parse_args())
    input_path = pathlib.Path(args['input_path'])
    name = args['name']
    if args['append']:
        df = pd.read_csv((input_path / name).with_suffix('.csv'), index_col=0)
        if 'conditioning' in df.columns:
            df = df.drop(columns=['conditioning'])
        if 'c-dry' in df.columns and args['force']:
            df = df.drop(columns=['c-dry', 'c-feedback_delay', 'c-slapback_delay',
                                      'c-reverb', 'c-chorus', 'c-flanger', 'c-phaser',
                                      'c-tremolo', 'c-vibrato', 'c-distortion', 'c-overdrive', 'fx_class'])
        columns = list(df.columns) + ['c-dry', 'c-feedback_delay', 'c-slapback_delay',
                                      'c-reverb', 'c-chorus', 'c-flanger', 'c-phaser',
                                      'c-tremolo', 'c-vibrato', 'c-distortion', 'c-overdrive', 'fx_class']
        df = df.reindex(columns=columns)
    else:
        df = pd.DataFrame(columns=['c-dry', 'c-feedback_delay', 'c-slapback_delay',
                                   'c-reverb', 'c-chorus', 'c-flanger', 'c-phaser',
                                   'c-tremolo', 'c-vibrato', 'c-distortion', 'c-overdrive', 'fx_class'])
    clf = torch.jit.load(args['model'])
    for file in tqdm.tqdm(input_path.rglob('*.wav')):
        audio, rate = torchaudio.load(input_path / file)
        # add noise to avoid NaNs
        audio += torch.randn_like(audio) * 1e-6
        if audio.shape[-1] < 44100 and args['padding']:
            to_pad = 44100 - audio.shape[-1]
            audio = F.pad(audio, (to_pad, 0))
            audio += torch.randn_like(audio) * 1e-6
        conditioning = clf(audio)
        fx_class = torch.argmax(conditioning)
        conditioning = conditioning.detach().numpy()
        fx_class = fx_class.detach().numpy() / 10
        df.loc[df["Unnamed: 0"] == file.stem, COLUMNS[:-1]] = conditioning[0]
        df.loc[df["Unnamed: 0"] == file.stem, 'fx_class'] = fx_class
    df.to_csv(input_path / 'data.csv')
    df.to_pickle(input_path / 'data.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Class conditioning.")
    parser.add_argument('--input-path', '-i', type=str or pathlib.Path,
                        help="Path to the processed sounds.")
    parser.add_argument('--model', '-m', type=str or pathlib.Path,
                        help="Path to compiled classifier.")
    parser.add_argument('--name', '-n', default='data', type=str,
                        help="Name to give to the output file.")
    parser.add_argument('--append', '-a', action='store_true')
    parser.add_argument('--padding', '-p', action='store_true')
    parser.add_argument('--force', '-f', action='store_true')
    sys.exit(main(parser))
