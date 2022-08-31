import argparse
import pathlib
import sys
from shutil import copy


def main(parser):
    args = vars(parser.parse_args())
    in_path = pathlib.Path(args['input_path'])
    out_path = pathlib.Path(args['output_path'])
    out_path.mkdir(parents=True, exist_ok=True)
    for f in in_path.rglob("Gitarre monophon/Samples/NoFX/*.wav"):
        f = pathlib.Path(f)
        if args['cut']:
            f.replace(out_path / f.name)
        else:
            copy(f, out_path / f.name)
    for f in in_path.rglob("Gitarre monophon/Samples/EQ/*.wav"):
        f = pathlib.Path(f)
        if args['cut']:
            f.replace(out_path / f.name)
        else:
            copy(f, out_path / f.name)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', '-i', type=str,
                        help="Path to the IDMT-SMT dataset")
    parser.add_argument('--output-path', '-o', type=str,
                        help="Folder where to copy clean files")
    parser.add_argument('--cut', '-c', action='store_true',
                        help="Remove copied files")
    sys.exit(main(parser))
