"""
Script for data augmentation
"""
import argparse
import pathlib
import sys
import pedalboard
from pedalboard.io import AudioFile
import soundfile
from tqdm import tqdm
import util
import pickle


def main(parser: argparse.ArgumentParser) -> int:
    args = vars(parser.parse_args())
    fx = []
    if args.get('chorus'):
        rate_hz = args['chorus_rate']
        depth = args['chorus_depth']
        cnt_delay = args['chorus_delay']
        feedback = args['chorus_feedback']
        mix = args['chorus_mix']
        fx.append(pedalboard.Chorus(rate_hz=rate_hz, depth=depth,
                                    centre_delay_ms=cnt_delay, feedback=feedback, mix=mix))
    if args.get('distortion'):
        drive = args['disto_drive']
        fx.append(pedalboard.Distortion(drive_db=drive))
    if args.get('reverb'):
        room = args['reverb_room_size']
        damping = args['reverb_damping']
        wet = args['reverb_wet']
        dry = args['reverb_dry']
        fx.append(pedalboard.Reverb(room_size=room, damping=damping, wet_level=wet, dry_level=dry))
    if len(fx) == 0 and 'board_path' not in args:
        raise ValueError("Choose at least one effect to apply to the dry audio.")
    if args['board_path'] is not None:
        with open(args['board_path'], 'rb') as f:
            board = pickle.load(f)
    else:
        board = pedalboard.Pedalboard(fx)
    in_path = pathlib.Path(args['in_path'])
    out_path = pathlib.Path(args['out_path'])
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
        for file in tqdm(in_path.iterdir()):
            audio, rate = util.read_audio(file)
            audio = util.apply_fx(audio, rate, board)
            out_name = file.stem + '.wav'
            soundfile.write(out_path / out_name, audio.T, int(rate))
        with open(out_path / 'fx.pkl', 'wb') as f:
            pickle.dump(board, f)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data augmentation by applying effects on dry sounds.")
    parser.add_argument('--chorus', action='store_true')
    parser.add_argument('--chorus-rate', type=float, default=1,
                        help="Rate in Hertz of the Chorus effect. Default is 1.")
    parser.add_argument('--chorus-depth', type=float, default=0.25,
                        help="LFO depth of the chorus effect. Default is 0.25. "
                             "It should be kept low for a standard chorus sound.")
    parser.add_argument('--chorus-delay', type=float, default=7,
                        help="LFO centre delay time in ms. Classic chorus sounds are obtained with a delay time around"
                             "7-8ms. Set it lower for a flanger effect.")
    parser.add_argument('--chorus-feedback', type=float, default=0,
                        help="Feedback volume. It should be low for a classic chorus sound and higher for a"
                             "flanger-like effect.")
    parser.add_argument('--chorus-mix', type=float, default=0.5,
                        help="Dry/wet mixing control. 0 is full dry (bypass) and 1 is full wet. Default is 0.5."
                             "A vibrato effect can be obtained if set to 1.")
    parser.add_argument('--distortion', action='store_true')
    parser.add_argument('--disto-drive', type=float, default=25,
                        help="Distortion's drive in dB. Default is 25.")
    parser.add_argument('--reverb', action='store_true')
    parser.add_argument('--reverb-room-size', type=float, default=0.5,
                        help="Reverb room-size between 0 and 1. The closer to one the longer the reverberation tail."
                             "Default is 0.5.")
    parser.add_argument('--reverb-damping', type=float, default=0.5,
                        help="Damping factor between 0 and 1. The closer to 0 the longer the reverberation tail."
                             "Default is 0.5.")
    parser.add_argument('--reverb-wet', type=float, default=0.33,
                        help="Wet signal level between 0 and 1. Default is 0.33.")
    parser.add_argument('--reverb-dry', type=float, default=0.4,
                        help="Dry signal level between 0 and 1. Default is 0.4.")
    parser.add_argument('--board-path', type=pathlib.Path or str, default=None,
                        help="Path to a file representing the Pedalboard.board to use. "
                             "As of now, only pickle files are supported.")
    parser.add_argument('--in-path', '-i', default=pathlib.Path('../source'), type=pathlib.Path,
                        help="Path to the dry sounds. Defaults to current directory.")
    parser.add_argument('--out-path', '-o', default=pathlib.Path('./out'), type=pathlib.Path,
                        help="Where to store the processed files. Defaults to ./out")
    parser.add_argument('--force', '-f', action='store_true')
    sys.exit(main(parser))
