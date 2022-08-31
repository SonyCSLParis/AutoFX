import pathlib
import argparse
from shutil import copy

from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Copy monophonic guitar files into a single folder.')
    parser.add_argument('--in-path', '-i', type=str,
                        help="Path to the complete IDMT dataset")
    parser.add_argument('--out-path', '-o', type=str,
                        help="Where to copy the files")
    parser.add_argument('--cut', '-c', action='store_true',
                        help="Should the original files be removed")
    args = vars(parser.parse_args())
    in_path = pathlib.Path(args['in_path'])
    out_path = pathlib.Path(args['out_path'])
    out_path.mkdir(parents=True, exist_ok=True)
    for f in tqdm.tqdm(in_path.rglob("*.wav")):
        f = pathlib.Path(f)
        if args['cut']:
            f.replace(out_path / f.name)
        else:
            copy(f, out_path / f.name)
