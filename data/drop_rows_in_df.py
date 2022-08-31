"""
Drop rows in dataframe if files are no longer in folder.
"""
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
    df = pd.read_csv(in_path / "data.csv", index_col=0)
    out_df = pd.DataFrame(columns=df.columns)
    for file in tqdm(in_path.rglob('*.wav')):
        out_df = out_df.append(df.loc[df["Unnamed: 0"] == file.stem, :], ignore_index=True)
    out_df.to_csv(in_path / 'data_cleaned.csv')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sort audio files according to their conditioning value.")
    parser.add_argument('--input-path', '-i', type=str or pathlib.Path,
                        help="Path to the input folder")
    sys.exit(main(parser))
