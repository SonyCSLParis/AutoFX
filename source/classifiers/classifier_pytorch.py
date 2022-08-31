import pathlib
import pickle
from typing import Any, Optional

import pandas as pd
import torch
import tqdm
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from ignite.metrics import Accuracy
from ignite.metrics.precision import Precision
from ignite.metrics.recall import Recall
from ignite.metrics.confusion_matrix import ConfusionMatrix
import torchaudio

import data.features_jit as Ft
import data.functional_jit as Fc
import source.util
import source.util as util


class TorchStandardScaler:
    """
    from    https://discuss.pytorch.org/t/pytorch-tensor-scaling/38576/8
    """

    def __init__(self):
        self.mean = 0
        self.std = 1

    def fit(self, x):
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)

    def transform(self, x):
        x -= self.mean
        x /= (self.std + 1e-7)
        return x


class ScalerModule(nn.Module):
    """
    from    https://discuss.pytorch.org/t/pytorch-tensor-scaling/38576/8
    """

    def __init__(self):
        super().__init__()
        self.mean = 0
        self.std = 1

    def fit(self, x):
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)

    def forward(self, x):
        x -= self.mean
        x /= (self.std + 1e-7)
        return x


class ClassificationDataset(Dataset):
    """
    Simple Dataset for iterating through samples to classify
    """

    def __init__(self, features, labels):
        super(ClassificationDataset, self).__init__()
        if isinstance(features, torch.Tensor):
            self.features = features
            self.labels = labels
        else:
            self.features = torch.tensor(features, dtype=torch.float)
            self.labels = torch.tensor(labels.values)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, item):
        return self.features[item], self.labels[item]


class MLPClassifier(pl.LightningModule):
    @staticmethod
    def _activation(activation):
        if activation == 'logistic' or activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise NotImplementedError

    @staticmethod
    def _to_one_hot(batch, num_classes):
        idx = torch.argmax(batch, dim=-1, keepdim=False)
        out = F.one_hot(idx, num_classes=num_classes)
        return out

    def __init__(self, input_size: int, output_size: int,
                 hidden_size: int, activation: str, solver: str,
                 max_iter: int, learning_rate: float = 0.0001,
                 tol: float = 1e-4, n_iter_no_change: int = 10):
        super(MLPClassifier, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.activation = MLPClassifier._activation(activation)
        self.solver = solver
        self.max_iter = max_iter
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.loss = nn.NLLLoss()
        self.softmax = nn.Softmax(dim=-1)
        self.prec = Precision()
        self.recall = Recall()
        self.accuracy = Accuracy()
        self.confusion_matrix = ConfusionMatrix(num_classes=output_size)
        self.learning_rate = learning_rate
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.save_hyperparameters()

    def forward(self, x) -> Any:
        out = self.linear1(x)
        out = self.activation(out)
        out = self.linear2(out)
        out = self.softmax(out)
        return out

    def training_step(self, batch, batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        feat, label = batch
        pred = self.forward(feat)
        loss = self.loss(torch.log(pred), label)
        self.log("loss/Train", loss)
        self.logger.experiment.add_scalar("Cross-entropy loss/Train", loss, global_step=self.global_step)
        classes = MLPClassifier._to_one_hot(pred, self.output_size)
        self.prec.reset()
        self.accuracy.reset()
        self.recall.reset()
        self.confusion_matrix.reset()
        self.prec.update((classes, label))
        self.accuracy.update((classes, label))
        self.recall.update((classes, label))
        # self.confusion_matrix.update((classes, label))
        precision = self.prec.compute()
        self.logger.experiment.add_scalars("Metrics/Precision_Train",
                                           dict(zip(util.CLASSES, precision)),
                                           global_step=self.global_step)
        accuracy = self.accuracy.compute()
        self.logger.experiment.add_scalar("Metrics/Accuracy_Train",
                                          accuracy,
                                          global_step=self.global_step)
        recall = self.recall.compute()
        self.logger.experiment.add_scalars("Metrics/Recall_Train",
                                           dict(zip(util.CLASSES, recall)),
                                           global_step=self.global_step)
        # confusion_matrix = self.confusion_matrix.compute()
        # fig = util.make_confusion_matrix(confusion_matrix.numpy(),
        #                                 group_names=CLASSES)
        # self.logger.experiment.add_figure("Metrics/Confusion_matrix",
        #                                  fig, global_step=self.global_step)
        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        feat, label = batch
        pred = self.forward(feat)
        loss = self.loss(torch.log(pred), label)
        self.log("loss/test", loss)
        self.logger.experiment.add_scalar("Cross-entropy loss/test", loss, global_step=self.global_step)
        classes = MLPClassifier._to_one_hot(pred, self.output_size)
        self.prec.reset()
        self.accuracy.reset()
        self.recall.reset()
        self.confusion_matrix.reset()
        self.prec.update((classes, label))
        self.accuracy.update((classes, label))
        self.recall.update((classes, label))
        # self.confusion_matrix.update((classes, label))
        precision = self.prec.compute()
        self.logger.experiment.add_scalars("Metrics/Precision_test",
                                           dict(zip(util.CLASSES, precision)),
                                           global_step=self.global_step)
        accuracy = self.accuracy.compute()
        self.logger.experiment.add_scalar("Metrics/Accuracy_test",
                                          accuracy,
                                          global_step=self.global_step)
        recall = self.recall.compute()
        self.logger.experiment.add_scalars("Metrics/Recall_test",
                                           dict(zip(util.CLASSES, recall)),
                                           global_step=self.global_step)

    def configure_optimizers(self):
        if self.solver == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            raise NotImplementedError
        return optimizer


class FeatureExtractor(nn.Module):
    @staticmethod
    def _apply_functionals(feat):
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
        pitch = pitch[:, None, None]
        cent = feat[:, :, :, 0]
        spread = feat[:, :, :, 1]
        skew = feat[:, :, :, 2]
        kurt = feat[:, :, :, 3]
        flux = feat[:, :, :, 4]
        rolloff = feat[:, :, :, 5]
        slope = feat[:, :, :, 6]
        flat = feat[:, :, :, 7]
        out = []
        out.append(FeatureExtractor._apply_functionals(cent))
        out.append(FeatureExtractor._apply_functionals(spread))
        out.append(FeatureExtractor._apply_functionals(skew))
        out.append(FeatureExtractor._apply_functionals(kurt))
        out.append(FeatureExtractor._apply_functionals(cent / pitch))
        out.append(FeatureExtractor._apply_functionals(spread / pitch))
        out.append(FeatureExtractor._apply_functionals(skew / pitch))
        # print("kurt/pitch", kurt / pitch)
        out.append(FeatureExtractor._apply_functionals(kurt / pitch))
        # print("flux", flux)
        out.append(FeatureExtractor._apply_functionals(flux))
        # print("rolloff", rolloff)
        # add some noise to avoid rolloff being constant
        rolloff = rolloff + torch.randn_like(rolloff) * 1e-4
        out.append(FeatureExtractor._apply_functionals(rolloff))
        out.append(FeatureExtractor._apply_functionals(slope))
        out.append(FeatureExtractor._apply_functionals(flat))
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
        out = torch.stack(out, dim=1)
        out = torch.reshape(out, (1, -1))
        out = torch.hstack([out[:, :52], out[:, 53:]])
        return out

    def __init__(self, rate: float = 22050, n_mfcc: int = 10):
        super(FeatureExtractor, self).__init__()
        self.spectrogram = torchaudio.transforms.Spectrogram(8192, hop_length=512, power=None)
        self.n_mfcc = n_mfcc
        self.mfcc_transform = torchaudio.transforms.MFCC(sample_rate=int(rate),
                                                         n_mfcc=n_mfcc)


    def forward(self, audio: torch.Tensor, rate: int = 22050):
        """

        :param audio: (batch, num_samples), mono only for now
        :param rate:
        :return:
        """
        # add some noise
        # audio = audio + torch.randn_like(audio) * (torch.max(torch.abs(audio)) / 1000)
        # if n_mfcc is None:
        #     n_mfcc = self.n_mfcc
        # if transform is None:
        #    transform = self.mfcc_transform
        mfcc = self.mfcc_transform(audio)
        mfcc_means = torch.mean(mfcc, dim=-1)
        mfcc_maxs = torch.max(mfcc, dim=-1)[0]
        # mfcc_means, mfcc_maxs = self.mfcc_torch(audio, rate, num_coeff=self.n_mfcc, mfcc_transform=self.mfcc_transform)
        stft = self.spectrogram(audio)
        mag = torch.abs(stft)
        feat = FeatureExtractor._get_features(mag, rate)
        pitch = Ft.pitch_curve(audio, rate)
        pitch = torch.mean(pitch, dim=-1)
        func = FeatureExtractor._get_functionals(feat, pitch)
        func = torch.cat([func, mfcc_means, mfcc_maxs], dim=1)
        return func

    def process_folder(self, folder_path: str, n_mfcc: int = 10, add_noise: bool = False):
        folder_path = pathlib.Path(folder_path)
        out = pd.DataFrame([])
        for f in tqdm.tqdm(folder_path.rglob('*.wav')):
            f = pathlib.Path(f)
            audio, rate = torchaudio.load(f)
            if add_noise:
                audio = audio + torch.randn_like(audio)*1e-6
            # print(f.name)
            if '_' not in f.name:
                fx = f.name.split('-')[2][1:-1]
                fx = util.idmt_fx2class_number(util.idmt_fx(fx))
            else:
                fx = f.name.split('_')[-1][:-4]
                # print(fx)
                match fx:
                    case 'distortion':
                        fx = 9
                    case 'modulation':
                        fx = 4
                    case 'delay':
                        fx = 2
            mfcc_transform = torchaudio.transforms.MFCC(sample_rate=rate,
                                                        n_mfcc=n_mfcc)
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
        print(out.shape)
        out.to_csv(folder_path / 'out.csv')
        out.to_pickle(folder_path / 'out.pkl')

    def mfcc_torch(audio, rate, num_coeff, mfcc_transform):
        # if transform is None:
        #    transform = torchaudio.transforms.MFCC(sample_rate=rate, n_mfcc=num_coeff)
        mfcc = mfcc_transform(audio)
        means = torch.mean(mfcc, dim=-1)
        maxs = torch.max(mfcc, dim=-1)[0]
        return means, maxs

class SilenceRemover(nn.Module):
    def __init__(self, start_threshold: float, end_threshold: float):
        super(SilenceRemover, self).__init__()
        self.start_thresh = start_threshold
        self.end_thresh = end_threshold

    def forward(self, audio):
        audio = audio / (torch.max(torch.abs(audio)))
        energy = torch.square(audio)
        start, end = util.find_attack_torch(energy, start_threshold=self.start_thresh, end_threshold=self.end_thresh)
        onset = int((start+end)/2)
        print(start, end, onset)
        return audio[:, onset:]

    def process_folder(self, in_path: str, out_path: str,
                       original_rate: int=44100, resampling_rate: int=None):
        in_path = pathlib.Path(in_path)
        out_path = pathlib.Path(out_path)
        if resampling_rate is not None:
            resampler = torchaudio.transforms.Resample(orig_freq=original_rate, new_freq=resampling_rate)
        else:
            resampler = None
            resampling_rate = original_rate
        for f in tqdm.tqdm(in_path.rglob('*.wav')):
            f = pathlib.Path(f)
            audio, rate = torchaudio.load(f, normalize=True)
            audio = resampler(audio)
            audio = self.forward(audio)
            torchaudio.save(out_path / f.name, audio, resampling_rate)
