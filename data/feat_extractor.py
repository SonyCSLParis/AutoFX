import argparse
import pathlib

import pandas as pd
import torch
import torchaudio
import tqdm
from torch import nn

import data.functional_jit as Fc
import data.features_jit as Ft
from source import util


class FeatureExtractor(nn.Module):
    """
    nn.Module to extract the features for classification of audio samples.
    """

    @staticmethod
    def _apply_functionals(feat):
        """
        Apply common functionals for most feature vectors:
        mean, std, skewness, kurtosis, min, max
        and stack them along last dimension.
        feat: (batch_size, 1, frames) or (batch_size, frames) array of the feature vectors
        return:
            (batch_size, 6) array of the scalars extracted from the feature vectors
        """
        if feat.ndim == 3:
            feat = feat[:, 0, :]
        out = []
        out.append(Fc.f_avg(feat, dim=1))
        out.append(Fc.f_std(feat, dim=1))
        out.append(Fc.f_skew(feat, dim=1))
        out.append(Fc.f_kurt(feat, dim=1))
        out.append(Fc.f_min(feat, dim=1))
        out.append(Fc.f_max(feat, dim=1))
        out = torch.stack(out, dim=-1)
        return out

    @staticmethod
    def _get_features(mag, rate: int):
        """
        Get spectral features from the magnitude spectra.
        :param mag: (batch_size, fft_size/2 + 1, num_frames) array of the magnitude spectra
        :param rate: sampling rate
        :return:
            (batch_size, 1, num_frames, 8) array of the frame-wise spectral features for each sample
        """
        centroid = Ft.spectral_centroid(mag=mag, rate=rate)
        spread = Ft.spectral_spread(mag=mag, cent=centroid, rate=rate)
        skew = Ft.spectral_skewness(mag=mag, cent=centroid, rate=rate)
        kurt = Ft.spectral_kurtosis(mag=mag, cent=centroid, rate=rate)
        flux = Ft.spectral_flux(mag=mag)
        rolloff = Ft.spectral_rolloff(mag=mag, rate=rate)
        slope = Ft.spectral_slope(mag=mag, rate=rate)
        flat = Ft.spectral_flatness(mag=mag, bands=1, rate=rate)
        out = torch.stack([centroid, spread, skew, kurt, flux,
                           rolloff, slope, flat], dim=-1)
        return out

    @staticmethod
    def _get_functionals(feat, pitch):
        """
        Prepare the feature vectors as presented in the report.
        :param feat: array of the spectral features obtained from _get_features
        :param pitch: (batch_size) pitch of the note played
        :return:
        """
        batch_size = feat.shape[0]
        # Fake expand of pitch
        pitch = pitch[:, None, None]
        # split spectral features
        cent = feat[:, :, :, 0]
        spread = feat[:, :, :, 1]
        skew = feat[:, :, :, 2]
        kurt = feat[:, :, :, 3]
        flux = feat[:, :, :, 4]
        rolloff = feat[:, :, :, 5]
        slope = feat[:, :, :, 6]
        flat = feat[:, :, :, 7]
        out = []
        # spectral moments
        out.append(FeatureExtractor._apply_functionals(cent))
        out.append(FeatureExtractor._apply_functionals(spread))
        out.append(FeatureExtractor._apply_functionals(skew))
        out.append(FeatureExtractor._apply_functionals(kurt))
        # pitch-normalized spectral moments
        out.append(FeatureExtractor._apply_functionals(cent / pitch))
        out.append(FeatureExtractor._apply_functionals(spread / pitch))
        out.append(FeatureExtractor._apply_functionals(skew / pitch))
        out.append(FeatureExtractor._apply_functionals(kurt / pitch))
        # other spectral features
        out.append(FeatureExtractor._apply_functionals(flux))
        # add noise to avoid NaNs due to zero variance
        rolloff = rolloff + torch.randn_like(rolloff) * 1e-4
        out.append(FeatureExtractor._apply_functionals(rolloff))
        out.append(FeatureExtractor._apply_functionals(slope))
        out.append(FeatureExtractor._apply_functionals(flat))
        # analyse delta features
        out.append(FeatureExtractor._apply_functionals(Fc.estim_derivative(cent, dim=-1)))
        out.append(FeatureExtractor._apply_functionals(Fc.estim_derivative(spread, dim=-1)))
        out.append(FeatureExtractor._apply_functionals(Fc.estim_derivative(skew, dim=-1)))
        out.append(FeatureExtractor._apply_functionals(Fc.estim_derivative(kurt, dim=-1)))
        out.append(FeatureExtractor._apply_functionals(Fc.estim_derivative(cent / pitch, dim=-1)))
        out.append(FeatureExtractor._apply_functionals(Fc.estim_derivative(spread / pitch, dim=-1)))
        out.append(FeatureExtractor._apply_functionals(Fc.estim_derivative(skew / pitch, dim=-1)))
        out.append(FeatureExtractor._apply_functionals(Fc.estim_derivative(kurt / pitch, dim=-1)))
        out.append(FeatureExtractor._apply_functionals(Fc.estim_derivative(flux, dim=-1)))
        out.append(FeatureExtractor._apply_functionals(Fc.estim_derivative(rolloff, dim=-1)))
        out.append(FeatureExtractor._apply_functionals(Fc.estim_derivative(slope, dim=-1)))
        out.append(FeatureExtractor._apply_functionals(Fc.estim_derivative(flat, dim=-1)))
        # stack features into a tensor
        out = torch.stack(out, dim=1)
        # make one long feature vector for each sample
        out = torch.reshape(out, (batch_size, -1))
        # remove min of spectral flux which is always zero
        out = torch.hstack([out[:, :52], out[:, 53:]])
        return out

    def __init__(self, rate: float = 22050, n_mfcc: int = 10):
        super(FeatureExtractor, self).__init__()
        # Store transforms to avoid instanciating one per file when processing a folder
        self.spectrogram = torchaudio.transforms.Spectrogram(8192, hop_length=512, power=None)
        self.n_mfcc = n_mfcc
        self.mfcc_transform = torchaudio.transforms.MFCC(sample_rate=int(rate),
                                                         n_mfcc=n_mfcc)

    def forward(self, audio: torch.Tensor, rate: int = 22050, add_noise: bool = False):
        """
        :param audio: (batch, num_samples), mono only for now
        :param rate (int): sampling rate in Hz. Default is 22050
        :param add_noise (bool): add noise during processing. Default is False
        :return:
        """
        # add some noise
        if add_noise:
            audio = audio + torch.randn_like(audio) * 1e-6
        mfcc = self.mfcc_transform(audio)
        mfcc_means = torch.mean(mfcc, dim=-1)
        mfcc_maxs = torch.max(mfcc, dim=-1)[0]
        stft = self.spectrogram(audio)
        mag = torch.abs(stft)
        feat = FeatureExtractor._get_features(mag, rate)
        pitch = Ft.pitch_curve(audio, rate)
        pitch = torch.mean(pitch, dim=-1)
        func = FeatureExtractor._get_functionals(feat, pitch)
        func = torch.cat([func, mfcc_means, mfcc_maxs], dim=1)
        return func

    def process_folder(self, folder_path: str, n_mfcc: int = 10, add_noise: bool = False,
                       rate: int = 22050, file_name: str = 'out'):
        """
        Extract classification features of all files inside a folder and store the results in a .csv file
        :param folder_path: absolute path to the folder
        :param n_mfcc: number of mfcc to compute. Default is 10.
        :param add_noise: Add noise during processing to avoid NaNs if necessary. Default is False.
        :param rate: Sampling rate of the audio files in Hz. Default is 22050.
        :param file_name: file_name to give to the output .csv file. Default is 'out'
        """
        if n_mfcc != self.n_mfcc:
            self.n_mfcc = n_mfcc
            self.mfcc_transform = torchaudio.transforms.MFCC(sample_rate=int(rate),
                                                             n_mfcc=n_mfcc)
        folder_path = pathlib.Path(folder_path)
        out = pd.DataFrame([])
        # Only try processing .wav files
        for f in tqdm.tqdm(folder_path.rglob('*.wav')):
            f = pathlib.Path(f)
            audio, rate = torchaudio.load(f)
            if add_noise:
                audio = audio + torch.randn_like(audio) * 1e-6
            fx = f.name.split('-')[2][1:-1]
            fx = util.idmt_fx2class_number(util.idmt_fx(fx))
            try:
                func = self.forward(audio, rate)
            except ValueError:
                print(f"One std was zero, this will return NaNs. File was {f}")
            if torch.isnan(func).any():
                raise ValueError(f"NaN returned while processing {f}: {func}")
            row = pd.DataFrame(func.numpy())
            row['file'] = f.name
            row['class'] = fx
            out = pd.concat([out, row], axis=0)
        file_name = pathlib.Path(file_name)
        out.to_csv(folder_path / file_name.with_suffix(".csv"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract the classification features from all .wav files of the input folder and write results in a .csv file.")
    parser.add_argument('--in', '-i', type=str,
                        help="Path to the sounds to process.")
    parser.add_argument('--noise', '-N', action='store_true',
                        help="Flag to add noise during processing to avoid NaNs if necessary.")
    parser.add_argument('--num-mfcc', '-n', type=int, default=10,
                        help="Number of MFCCs to compute. Default is 10.")
    parser.add_argument('--rate', '-r', type=int, default=22050,
                        help="Sampling rate of the sound files, in Hertz. Default is 22050 Hz.")
    parser.add_argument('--filename', '-F', type=str, default='out',
                        help="Name to give to the output .csv file. Default is 'out'.")
    args = vars(parser.parse_args())
    extractor = FeatureExtractor()
    extractor.process_folder(args['in'], n_mfcc=args['num_mfcc'], rate=args['rate'], add_noise=args['noise'],
                             file_name=args['filename'])