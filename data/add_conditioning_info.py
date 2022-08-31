import argparse
import pathlib
import sys

import torch
import torchaudio
import pandas as pd
import tqdm
import source.util as util


def main(parser):
    args = vars(parser.parse_args())
    in_path = pathlib.Path(args['input_path'])
    if args['append']:
        df = pd.read_csv(in_path / "data.csv", index_col=0)
        df['conditioning'] = None
    else:
        df = pd.DataFrame(columns=['conditioning'])
    for file in tqdm.tqdm(in_path.rglob('*.wav')):
        file = pathlib.Path(file)
        fx = file.stem.split('-')[2][1:3]
        conditioning = util.idmt_fx2class_number(util.idmt_fx(fx))
        conditioning = conditioning / 10
        df.loc[file.stem, 'conditioning'] = conditioning
    df.to_csv(in_path / 'data.csv')
    df.to_pickle(in_path / 'data.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Class conditioning.")
    parser.add_argument('--input-path', '-i', type=str or pathlib.Path,
                        help="Path to the processed sounds.")
    parser.add_argument('--name', '-n', default='data', type=str,
                        help="Name to give to the output file")
    parser.add_argument('--append', '-a', action='store_true')
    sys.exit(main(parser))
