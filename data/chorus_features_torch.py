import argparse
import pathlib
import sys

import numpy as np
import torch
import torchaudio
import soundfile as sf
import data.features as Ft
import data.functional as Fc
from tqdm import tqdm
from data import superflux
import pandas as pd

import source.util as util

FEATURES = ['f-phase_fft_max', 'f-phase_freq', 'f-rms_fft_max', 'f-rms_freq',
            'f-phase_fft_max2', 'f-phase_freq2', 'f-rms_fft_max2', 'f-rms_freq2',
            'f-rms_delta_fft_max', 'f-rms_delta_freq', 'f-rms_delta_fft_max2', 'f-rms_delta_freq2',
            'f-phase_delta_fft_max', 'f-phase_delta_freq', 'f-phase_delta_fft_max2', 'f-phase_delta_freq2',
            'f-pitch_fft_max', 'f-pitch_freq', 'f-pitch_delta_fft_max', 'f-pitch_delta_freq',
            'f-pitch_fft_max2', 'f-pitch_freq2', 'f-pitch_delta_fft_max2', 'f-pitch_delta_freq2',
            'f-rms_std', 'f-rms_delta_std', 'f-rms_skew', 'f-rms_delta_skew',
            'f-onset0', 'f-onset1', 'f-onset2',  'f-onset3', 'f-onset4',
            'f-activation0', 'f-activation1', 'f-activation2',
            'f-activation3', 'f-activation4',
            'f-mfcc0', 'f-mfcc1', 'f-mfcc2', 'f-mfcc3', 'f-mfcc4',
            'f-mfcc5', 'f-mfcc6', 'f-mfcc7', 'f-mfcc8', 'f-mfcc9'
            ]

FEAT2APP = ['f-mfcc0', 'f-mfcc1', 'f-mfcc2', 'f-mfcc3', 'f-mfcc4',
            'f-mfcc5', 'f-mfcc6', 'f-mfcc7', 'f-mfcc8', 'f-mfcc9']


def main(parser):
    args = vars(parser.parse_args())
    in_path = pathlib.Path(args['input_path'])
    out_path = pathlib.Path(args['output_path'])
    out_csv = out_path / (args['name'] + '.csv')
    out_pkl = out_path / (args['name'] + '.pkl')
    filt = superflux.Filter(2048 // 2 + 1, rate=22050, bands=24, fmin=30, fmax=17000, equal=False)
    filterbank = filt.filterbank
    transform = torchaudio.transforms.MFCC(sample_rate=22050, n_mfcc=10)
    if (in_path / "params.csv").exists():
        df = pd.read_csv(in_path / "params.csv")
    else:
        df = pd.DataFrame()
    if args['append']:
        df = pd.read_csv(out_csv, index_col=None)
        df[FEAT2APP] = np.nan
    else:
        df[FEATURES] = np.nan
    if not in_path.is_absolute():
        in_path = pathlib.Path.cwd() / in_path
    if not out_path.is_absolute():
        out_path = pathlib.Path.cwd() / out_path
    if not args['force'] and not args['append']:
        if out_csv.exists() or out_pkl.exists():
            raise FileExistsError("Output directory is not empty. Add --force or -f to overwrite anyway.")
    if not out_path.exists():
        out_path.mkdir()
    if not in_path.exists():
        raise FileNotFoundError("Input directory cannot be found.")
    else:
        if not in_path.is_dir():
            raise NotADirectoryError("Input path is not a directory.")
        for file in tqdm(in_path.iterdir()):
            file = pathlib.Path(file)
            if file.suffix == '.wav':
                audio, rate = torchaudio.load(file)
                # normalize
                audio = audio / torch.max(torch.abs(audio))
                # add some noise
                audio = audio + torch.randn_like(audio) * 1e-9
                if args['append']:
                    mfccs = transform(audio)
                    mfccs = torch.mean(mfccs[0], dim=-1)
                    feat2add = mfccs.detach().numpy().tolist()
                    df.loc[df['Unnamed: 0'] == file.stem,
                           FEAT2APP] = feat2add
                else:
                    midi_pitch = int(file.name.split('-')[1][:2])
                    hz_pitch = util.midi2hertz(midi_pitch)
                    phase = Ft.phase_fmax_batch(audio)
                    rms = Ft.rms_energy(audio, torch_compat=True)
                    rms_delta = Fc.estim_derivative(rms, torch_compat=True)
                    phase_delta = Fc.estim_derivative(phase, torch_compat=True)
                    low_freq = max(50, hz_pitch - 200)
                    high_freq = min(rate, hz_pitch + 200)
                    pitch = Ft.pitch_curve(audio, rate, low_freq, high_freq, hz_pitch, torch_compat=True)
                    pitch = pitch
                    pitch_delta = Fc.estim_derivative(pitch, torch_compat=True)
                    pitch_fft_max, pitch_freq = Fc.fft_max_batch(pitch,
                                                           num_max=2,
                                                           zero_half_width=32)
                    pitch_delta_fft_max, pitch_delta_freq = Fc.fft_max_batch(pitch_delta,
                                                                       num_max=2,
                                                                       zero_half_width=32)
                    rms_delta_fft_max, rms_delta_freq = Fc.fft_max_batch(rms_delta,
                                                                   num_max=2,
                                                                   zero_half_width=32)
                    phase_delta_fft_max, phase_delta_freq = Fc.fft_max_batch(phase_delta,
                                                                       num_max=2,
                                                                       zero_half_width=32)
                    phase_fft_max, phase_freq = Fc.fft_max_batch(phase, num_max=2, zero_half_width=32)
                    rms_fft_max, rms_freq = Fc.fft_max_batch(rms, num_max=2, zero_half_width=32)
                    rms_std = Fc.f_std(rms, torch_compat=True)
                    rms_skew = Fc.f_skew(rms[0], torch_compat=True)
                    rms_delta_std = Fc.f_std(rms_delta, torch_compat=True)
                    rms_delta_skew = Fc.f_skew(rms_delta[0], torch_compat=True)
                    mfccs = transform(audio)
                    mfccs = torch.mean(mfccs[0], dim=-1)
                    features = [phase_fft_max[0, 0].item(), phase_freq[0, 0].item() / 512,
                                rms_fft_max[0, 0].item(), rms_freq[0, 0].item() / 512,
                                phase_fft_max[0, 1].item(), phase_freq[0, 1].item() / 512,
                                rms_fft_max[0, 1].item(), rms_freq[0, 1].item() / 512,
                                rms_delta_fft_max[0, 0].item(), rms_delta_freq[0, 0].item() / 512,
                                rms_delta_fft_max[0, 1].item(), rms_delta_freq[0, 1].item() / 512,
                                phase_delta_fft_max[0, 0].item(), phase_delta_freq[0, 0].item() / 512,
                                phase_delta_fft_max[0, 1].item(), phase_delta_freq[0, 1].item() / 512,
                                pitch_delta_fft_max[0, 0].item(), pitch_delta_freq[0, 0].item() / 512,
                                pitch_delta_fft_max[0, 1].item(), pitch_delta_freq[0, 1].item() / 512,
                                pitch_fft_max[0, 0].item(), pitch_freq[0, 0].item() / 512,
                                pitch_fft_max[0, 1].item(), pitch_freq[0, 1].item() / 512,
                                rms_std.item(), rms_delta_std.item(),
                                rms_skew.item(), rms_delta_skew.item()
                                ]
                    onsets, activations = Ft.onset_detection(audio, rate, filterbank)
                    features = features + onsets[0].detach().numpy().tolist()
                    features = features + activations[0].detach().numpy().tolist()
                    features = features + mfccs.detach().numpy().tolist()
                    if 'Unnamed: 0' in df.columns:
                        df.loc[df['Unnamed: 0'] == file.stem,
                               FEATURES] = features
                    else:
                        df.loc[file.stem] = features
    df.to_csv(out_csv)
    df.to_pickle(out_pkl)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Computing features of audio files for a chorus effect.")
    parser.add_argument('--input-path', '-i', type=str or pathlib.Path,
                        help="Path to the processed sounds.")
    parser.add_argument('--output-path', '-o', type=str or pathlib.Path,
                        help="Path to an output folder to store the analysis results.")
    parser.add_argument('--name', '-n', default='data', type=str,
                        help="Name to give to the output file")
    parser.add_argument('--append', '-a', action='store_true')
    parser.add_argument('--force', '-f', action='store_true')
    sys.exit(main(parser))
