import os
import sys
import shutil

import pandas as pd
import pathlib
import argparse

from tqdm import tqdm


def main(parser):
    args = vars(parser.parse_args())
    in_path = pathlib.Path(args['input_path'])
    out_path = pathlib.Path(args['output_path'])
    df = pd.read_csv(in_path / "data.csv")
    for file in tqdm(in_path.rglob('*.wav')):
        conditioning = df.loc[df["Unnamed: 0"] == file.stem, 'conditioning']
        conditioning = str(conditioning.item())
        if not (out_path / conditioning).exists():
            os.mkdir(out_path / conditioning)
        shutil.copy(file, out_path / conditioning)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sort audio files according to their conditioning value.")
    parser.add_argument('--input-path', '-i', type=str or pathlib.Path,
                        help="Path to the processed sounds.")
    parser.add_argument('--output-path', '-o', type=str or pathlib.Path,
                        help="Path where to store sorted files")
    sys.exit(main(parser))
