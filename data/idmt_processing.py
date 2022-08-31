"""
Script for processing the IDMT-SMT dataset and saving the resulting features.
"""
import argparse
import pathlib
import sys

from tqdm import tqdm

import source.config as c
from soundsample import SoundSample
import pickle


def main(parser: argparse.ArgumentParser):
    args = vars(parser.parse_args())
    in_path = pathlib.Path(args['input_path'])
    out_path = pathlib.Path(args['output_path'])
    if not in_path.is_absolute():
        in_path = pathlib.Path.cwd() / in_path
    if not out_path.is_absolute():
        out_path = pathlib.Path.cwd() / out_path
    if out_path.exists():
        if not out_path.is_dir():
            raise NotADirectoryError("Output path is not a directory.")
        elif any(out_path.iterdir()) and not args['force']:
            raise FileExistsError("Output directory is not empty. Add --force or -f to overwrite anyway.")
    else:
        out_path.mkdir()
    if not in_path.exists():
        raise FileNotFoundError("Input directory cannot be found.")
    else:
        if not in_path.is_dir():
            raise NotADirectoryError("Input path is not a directory.")
        for child in tqdm(in_path.iterdir()):
            child = pathlib.Path(child)
            if child.is_dir():
                child_out = out_path / child.name
                child_out.mkdir()
                for file in tqdm(child.iterdir()):
                    sample = SoundSample(file, idmt=True)
                    sample.set_stft(fft_size=args['fft_size'])
                    sample.set_spectral_features(flux_q_norm=args['q_norm'])
                    with open(pathlib.Path(child_out / sample.file.name).with_suffix('.pkl'), 'wb') as f:
                        pickle.dump(sample.spectral_features, f)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analysis of the IDMT-SMT dataset and saving of all features.")
    parser.add_argument('--input-path', '-i', type=str or pathlib.Path,
                        help="Path to the IDMT-SMT dataset.")
    parser.add_argument('--output-path', '-o', type=str or pathlib.Path,
                        help="Path to an output folder to store the analysis results.")
    parser.add_argument('--fft-size', '-N', default=c.FFT_SIZE, type=int,
                        help="Size of the FFT to compute the STFT of each sound.")
    parser.add_argument('--q-norm', '-Q', default=c.Q_NORM, type=int,
                        help="Q norm to use for the spectral flux.")
    parser.add_argument('--force', '-f', action='store_true')
    sys.exit(main(parser))
