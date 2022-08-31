import argparse
import pathlib
import sys

import numpy as np
import soundfile as sf
import data.features as Ft
import data.functional as Fc
from tqdm import tqdm
import pandas as pd

import source.util as util

FEATURES = ['f-phase_fft_max', 'f-phase_freq', 'f-rms_fft_max', 'f-rms_freq',
            'f-phase_fft_max2', 'f-phase_freq2', 'f-rms_fft_max2', 'f-rms_freq2',
            'f-rms_delta_fft_max', 'f-rms_delta_freq', 'f-rms_delta_fft_max2', 'f-rms_delta_freq2',
            'f-phase_delta_fft_max', 'f-phase_delta_freq', 'f-phase_delta_fft_max2', 'f-phase_delta_freq2',
            'f-pitch_fft_max', 'f-pitch_freq', 'f-pitch_delta_fft_max', 'f-pitch_delta_freq',
            'f-pitch_fft_max2', 'f-pitch_freq2', 'f-pitch_delta_fft_max2', 'f-pitch_delta_freq2',
            'f-rms_std', 'f-rms_delta_std', 'f-rms_skew', 'f-rms_delta_skew'
            ]

FEAT2APP = ['f-rms_std', 'f-rms_delta_std', 'f-rms_skew', 'f-rms_delta_skew']


def main(parser):
    args = vars(parser.parse_args())
    in_path = pathlib.Path(args['input_path'])
    out_path = pathlib.Path(args['output_path'])
    out_csv = out_path / (args['name'] + '.csv')
    out_pkl = out_path / (args['name'] + '.pkl')
    if (in_path / "params.csv").exists():
        df = pd.read_csv(in_path / "params.csv")
    else:
        df = pd.DataFrame()
    if args['append']:
        df = pd.read_csv(out_csv, index_col=0)
        df[FEAT2APP] = np.nan
    else:
        df[FEATURES] = np.nan
    if not in_path.is_absolute():
        in_path = pathlib.Path.cwd() / in_path
    if not out_path.is_absolute():
        out_path = pathlib.Path.cwd() / out_path
    if not args['force']:
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
                audio, rate = sf.read(file)
                if args['append']:
                    rms = Ft.rms_energy(audio)
                    rms_delta = Fc.estim_derivative(rms)
                    rms_std = Fc.f_std(rms)
                    rms_skew = Fc.f_skew(rms[0])
                    rms_delta_std = Fc.f_std(rms_delta)
                    rms_delta_skew = Fc.f_skew(rms_delta[0])
                    feat2add = [rms_std, rms_delta_std,
                                rms_skew, rms_delta_skew]
                    df.loc[df['Unnamed: 0'] == file.stem,
                           FEAT2APP] = feat2add
                else:
                    midi_pitch = int(file.name.split('-')[1][:2])
                    hz_pitch = util.midi2hertz(midi_pitch)
                    phase = Ft.phase_fmax(audio)
                    rms = Ft.rms_energy(audio)
                    rms_delta = Fc.estim_derivative(rms)
                    phase_delta = Fc.estim_derivative(phase)
                    low_freq = max(50, hz_pitch - 200)
                    high_freq = min(rate, hz_pitch + 200)
                    pitch = Ft.pitch_curve(audio, rate, low_freq, high_freq, hz_pitch)
                    pitch = pitch[None, :]
                    pitch_delta = Fc.estim_derivative(pitch)
                    pitch_fft_max, pitch_freq = Fc.fft_max(pitch,
                                                           num_max=2,
                                                           zero_half_width=32)
                    pitch_delta_fft_max, pitch_delta_freq = Fc.fft_max(pitch_delta,
                                                                       num_max=2,
                                                                       zero_half_width=32)
                    rms_delta_fft_max, rms_delta_freq = Fc.fft_max(rms_delta,
                                                                   num_max=2,
                                                                   zero_half_width=32)
                    phase_delta_fft_max, phase_delta_freq = Fc.fft_max(phase_delta,
                                                                       num_max=2,
                                                                       zero_half_width=32)
                    phase_fft_max, phase_freq = Fc.fft_max(phase, num_max=2, zero_half_width=32)
                    rms_fft_max, rms_freq = Fc.fft_max(rms, num_max=2, zero_half_width=32)
                    rms_std = Fc.f_std(rms)
                    rms_skew = Fc.f_skew(rms[0])
                    rms_delta_std = Fc.f_std(rms_delta)
                    rms_delta_skew = Fc.f_skew(rms_delta[0])
                    features = [phase_fft_max[0], phase_freq[0] / 512,
                                rms_fft_max[0], rms_freq[0] / 512,
                                phase_fft_max[1], phase_freq[1] / 512,
                                rms_fft_max[1], rms_freq[1] / 512,
                                rms_delta_fft_max[0], rms_delta_freq[0] / 512,
                                rms_delta_fft_max[1], rms_delta_freq[1] / 512,
                                phase_delta_fft_max[0], phase_delta_freq[0] / 512,
                                phase_delta_fft_max[1], phase_delta_freq[1] / 512,
                                pitch_delta_fft_max[0], pitch_delta_freq[0] / 512,
                                pitch_delta_fft_max[1], pitch_delta_freq[1] / 512,
                                pitch_fft_max[0], pitch_freq[0] / 512,
                                pitch_fft_max[1], pitch_freq[1] / 512,
                                rms_std, rms_delta_std,
                                rms_skew, rms_delta_skew
                                ]
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
