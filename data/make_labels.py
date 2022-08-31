"""
Script to generate labels csv file for a dataset of soundfiles
"""

import argparse
import os
import pathlib
import pandas as pd
import soundsample
import sys
from tqdm import tqdm

def main(parser: argparse.ArgumentParser):
    args = vars(parser.parse_args())
    in_path = pathlib.Path(args['input'])
    out_path = pathlib.Path(args['output'])
    if not in_path.exists() or not in_path.is_dir():
        raise ValueError("Input path does not exist or is not a directory.")
    if out_path.is_dir():
        out_path = out_path / 'labels.csv'
    if out_path.exists() and not args['force']:
        raise FileExistsError("Add --force (-f) if you want to overwrite existing file.")
    if args['idmt']:
        df = pd.DataFrame(columns=['fx_type'])
        sizecounter = 0
        for filepath in in_path.glob('**/*.wav'):
            sizecounter += os.stat(filepath).st_size
        with tqdm(total=sizecounter,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            for child in in_path.iterdir():
                if child.is_dir():
                    for file in child.iterdir():
                        if file.suffix == '.wav':
                            pbar.set_postfix(file=file.stem[-5:], refresh=False)
                            info = soundsample.SoundSample.idmt_parsing(file.stem)
                            df.loc[info['id']] = info['fx']
                            pbar.update(os.stat(file).st_size)
    df.to_csv(out_path)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Make labels from sound dataset")
    parser.add_argument('--input', '-i', type=str or pathlib.Path,
                        help="Path to the sounds to generate the labels from.")
    parser.add_argument('--output', '-o', type=str or pathlib.Path,
                        help="Where to save the output file.")
    parser.add_argument('--idmt', action='store_true',
                        help="Add this argument if the sounds are from the IDMT-SMT dataset.")
    parser.add_argument('--force', '-f', action='store_true')
    sys.exit(main(parser))
