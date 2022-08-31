"""
Script to make the dataset for the SVM classifier
"""

import argparse
import pathlib
import sys

from tqdm import tqdm

import source.config as c
import pandas as pd

import source.util as util
import functional
from soundsample import SoundSample
import pickle


def main(parser: argparse.ArgumentParser):
    dframe = pd.DataFrame(columns=c.DATA_DICT.keys())
    dframe['target_name'] = None
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
    if not in_path.exists():
        raise FileNotFoundError("Input directory cannot be found.")
    else:
        if not in_path.is_dir():
            raise NotADirectoryError("Input path is not a directory.")
        for child in tqdm(in_path.iterdir()):
            child = pathlib.Path(child)
            if child.is_dir():
                for file in tqdm(child.iterdir()):
                    with open(file, 'rb') as f:
                        data = pickle.load(f)
                    info = SoundSample.idmt_parsing(pathlib.Path(file).stem)
                    pitch = util.midi2hertz(int(info['midi_pitch']))
                    func = functional.feat_vector(data, pitch)
                    func['target_name'] = info['fx']
                    dframe.loc[info['id']] = func
    out_csv = out_path / 'dataset.csv'
    out_pkl = out_path / 'dataset.pkl'
    dframe.to_csv(out_csv)
    dframe.to_pickle(out_pkl)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Computing all the functionals of the spectral features")
    parser.add_argument('--input-path', '-i', type=str or pathlib.Path,
                        help="Path to the processed IDMT-SMT dataset.")
    parser.add_argument('--output-path', '-o', type=str or pathlib.Path,
                        help="Path to an output folder to store the dataset.")
    parser.add_argument('--force', '-f', action='store_true')
    sys.exit(main(parser))
