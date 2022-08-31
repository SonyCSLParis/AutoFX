"""
Script for processing the IDMT-SMT dataset and saving the resulting features.
"""
import argparse
import pathlib
import sys

import pandas as pd
from tqdm import tqdm

import config as c
import functional
import util
from soundsample import SoundSample
import pickle


def main(parser: argparse.ArgumentParser):
    dframe = pd.DataFrame(columns=c.DATA_DICT.keys())
    dframe['target_name'] = None
    args = vars(parser.parse_args())
    in_path = pathlib.Path(args['input_path'])
    out_path = pathlib.Path(args['output_path'])
    out_csv = out_path / (args['name'] + '.csv')
    out_pkl = out_path / (args['name'] + '.pkl')
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
        d = 0
        f = 0
        for child in tqdm(in_path.iterdir()):
            child = pathlib.Path(child)
            if child.is_dir():
                d += 1
                f = 0
                for file in tqdm(child.iterdir()):
                    file = pathlib.Path(file)
                    if file.suffix == '.wav':
                        f += 1
                        sample = SoundSample(file, idmt=args['idmt'], phil=args['phil'])
                        sample.set_stft(fft_size=args['fft_size'])
                        sample.set_spectral_features(flux_q_norm=args['q_norm'])
                        func = functional.feat_vector(sample.spectral_features, sample.pitch)
                        # func['target_name'] = sample.fx
                        if sample.id is None:
                            sample.info['id'] = int(str(d) + str(f))
                        dframe.loc[str(sample.id) + child.name] = func
            elif child.suffix == '.wav':
                file = child
                f += 1
                sample = SoundSample(file, idmt=args['idmt'], phil=args['phil'])
                sample.set_stft(fft_size=args['fft_size'])
                sample.set_spectral_features(flux_q_norm=args['q_norm'])
                func = functional.feat_vector(sample.spectral_features, sample.pitch)
                # func['target_name'] = sample.fx
                if sample.id is None:
                    sample.info['id'] = int(str(d) + str(f))
                dframe.loc[str(sample.id) + child.name] = func
    dframe.to_csv(out_csv)
    dframe.to_pickle(out_pkl)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analysis of the IDMT-SMT dataset and saving of all features.")
    parser.add_argument('--input-path', '-i', type=str or pathlib.Path,
                        help="Path to the IDMT-SMT dataset.")
    parser.add_argument('--output-path', '-o', type=str or pathlib.Path,
                        help="Path to an output folder to store the analysis results.")
    parser.add_argument('--name', '-n', default='dataset', type=str,
                        help="Name to give to the output files")
    parser.add_argument('--fft-size', '-N', default=c.FFT_SIZE, type=int,
                        help="Size of the FFT to compute the STFT of each sound.")
    parser.add_argument('--idmt', action='store_true',
                        help="Set if data is from the IDMT dataset")
    parser.add_argument('--phil', action='store_true',
                        help="Set if data is from the Philarmonia dataset")
    parser.add_argument('--q-norm', '-Q', default=c.Q_NORM, type=int,
                        help="Q norm to use for the spectral flux.")
    parser.add_argument('--force', '-f', action='store_true')
    sys.exit(main(parser))
