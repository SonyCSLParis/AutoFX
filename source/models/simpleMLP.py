import os
import pathlib
from typing import Any, Optional

import auraloss
import numpy as np
import pedalboard as pdb
import pickle
import pytorch_lightning as pl
import sklearn.preprocessing
import torch
import torchaudio
from carbontracker.tracker import CarbonTracker
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import Dataset, DataLoader

import data.functional as Fc
import data.features as Ft
from data import superflux
from source.classifiers.classifier_pytorch import TorchStandardScaler
from source.data.datamodules import FeaturesDataModule
from source.models.custom_distortion import CustomDistortion
from source.models.mbfx_layer import MBFxLayer
from source.multiband_fx import MultiBandFX


class SimpleMLP(pl.LightningModule):
    """A simple MLP trained directly on features."""
    def __init__(self, num_features, hidden_size, num_hidden_layers,
                 scaler_mean, scaler_std,
                 param_range_modulation, param_range_delay, param_range_disto,
                 rate: float = 22050, learning_rate: float = 0.001,
                 mrstft_fft: list[int] = [64, 128, 256, 512, 1024, 2048],
                 mrstft_hop: list[int] = [16, 32, 64, 128, 256, 512],
                 monitor_spectral_loss: bool = False, audiologs: int = 8,
                 tracker: bool = False):
        super(SimpleMLP, self).__init__()
        num_bands = 1
        total_num_bands = 1
        self.num_bands = 1
        self.tracker_flag = tracker
        self.tracker = None
        self.total_num_bands = 1
        self.rate = rate
        self.num_features = num_features
        self.num_hidden_layers = num_hidden_layers
        if self.num_hidden_layers > 1:
            self.hidden_layers = nn.ModuleList(
                [nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU()) for i in range(num_hidden_layers - 1)])
        self.hidden_size = hidden_size
        modulation = MultiBandFX([pdb.Chorus], total_num_bands, device=torch.device('cpu'))
        delay = MultiBandFX([pdb.Delay], total_num_bands, device=torch.device('cpu'))
        disto = CustomDistortion()
        self.param_range_modulation = param_range_modulation
        self.param_range_delay = param_range_delay
        self.param_range_disto = param_range_disto
        self.num_params = num_bands * modulation.total_num_params_per_band \
                          + num_bands * delay.total_num_params_per_band \
                          + disto.total_num_params_per_band
        self.board = [modulation, delay, disto]
        delay_layer = MBFxLayer(self.board[1], self.rate, self.param_range_delay, fake_num_bands=self.num_bands)
        modulation_layer = MBFxLayer(self.board[0], self.rate, self.param_range_modulation,
                                     fake_num_bands=self.num_bands)
        disto_layer = MBFxLayer(self.board[2], self.rate, self.param_range_disto, fake_num_bands=self.num_bands)
        self.board_layers = [modulation_layer, delay_layer, disto_layer]
        self.scaler = TorchStandardScaler()
        self.scaler.mean = torch.tensor(scaler_mean, device=torch.device('cuda'))
        print(self.scaler.mean.shape)
        self.scaler.std = torch.tensor(scaler_std, device=torch.device('cuda'))
        self.fcl_start = nn.Linear(num_features + 6, hidden_size)
        self.fcl_end = nn.Linear(hidden_size, self.num_params)
        self.batchnorm = nn.BatchNorm1d(hidden_size, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.loss = nn.MSELoss()
        self.mrstft = auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=mrstft_fft,
                                                            hop_sizes=mrstft_hop,
                                                            win_lengths=mrstft_fft,
                                                            w_phs=1,
                                                            w_sc=0)
        self.spectral_loss = self.mrstft
        filt = superflux.Filter(2048 // 2 + 1, rate=22050, bands=24, fmin=30, fmax=17000, equal=False)
        self.filterbank = filt.filterbank.to('cuda')
        self.mfcc_transform = torchaudio.transforms.MFCC(n_mfcc=10, sample_rate=int(rate))
        self.feature_spectro = torchaudio.transforms.Spectrogram(n_fft=2048, hop_length=256, power=None)
        self.activation = nn.Sigmoid()
        self.monitor_spectral_loss = monitor_spectral_loss
        self.audiologs = audiologs
        self.feat_weight = 0
        self.mrstft_weight = 1
        self.out_of_domain = False
        self.learning_rate = learning_rate
        self.save_hyperparameters()

    def on_train_epoch_start(self) -> None:
        if self.tracker_flag and self.tracker is None:
            self.tracker = CarbonTracker(epochs=20, epochs_before_pred=1, monitor_epochs=1,
                                         log_dir=self.logger.log_dir, verbose=2)  # TODO: Remove hardcoded values
        if self.tracker_flag:
            self.tracker.epoch_start()

    def on_train_epoch_end(self) -> None:
        if self.tracker_flag:
            self.tracker.epoch_end()

    def forward(self, feat, conditioning, *args, **kwargs) -> Any:
        if conditioning[0] != 'None':
            x = torch.cat([feat, conditioning], dim=-1)
        else:
            x = feat
        out = self.fcl_start(x)
        if self.num_hidden_layers > 1:
            for l in self.hidden_layers:
                out = l(out)
        else:
            out = self.relu(out)
        out = self.fcl_end(out)
        out = self.activation(out)
        return out

    def training_step(self, batch, batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        clean, processed, feat, label, conditioning, fx_class = batch
        batch_size = processed.shape[0]
        pred = self.forward(feat, conditioning)
        pred_clone = pred.clone()
        # mask predictions according to fx_class
        pred_clone[:, :5] *= (fx_class[:, None] == 0)
        label[:, :5] *= (fx_class[:, None] == 0)
        pred_clone[:, 5:8] *= (fx_class[:, None] == 1)
        label[:, 5:8] *= (fx_class[:, None] == 1)
        pred_clone[:, 8:] *= (fx_class[:, None] == 2)
        label[:, 8:] *= (fx_class[:, None] == 2)
        loss = self.loss(pred_clone, label)
        pred_per_fx = [pred[:, :5], pred[:, 5:8], pred[:, 8:]]
        # loss = 0
        self.logger.experiment.add_scalar("Param_loss/Train", loss, global_step=self.global_step)
        scalars = {}
        for (i, val) in enumerate(torch.mean(torch.abs(pred_clone - label), 0)):
            scalars[f'{i}'] = val
        self.logger.experiment.add_scalars("Param_distance/Train", scalars, global_step=self.global_step)
        if self.monitor_spectral_loss:
            pred_per_fx = [ppf.to("cpu") for ppf in pred_per_fx]
            # self.pred = pred
            # self.pred.retain_grad()
            rec = torch.zeros(batch_size, clean.shape[-1], device=self.device)
            for (i, snd) in enumerate(clean):
                snd_norm = snd / torch.max(torch.abs(snd))
                snd_norm = snd_norm + torch.randn_like(snd_norm) * 1e-9
                tmp = self.board_layers[fx_class[i]].forward(snd_norm.cpu(), pred_per_fx[fx_class[i]][i])
                rec[i] = tmp.clone() / torch.max(torch.abs(tmp))
            target_normalized = processed[:, 0, :] / torch.max(torch.abs(processed[:, 0, :]), dim=-1, keepdim=True)[0]
            pred_normalized = rec.to(self.device)
            spec_loss = self.spectral_loss(pred_normalized, target_normalized)
            # features = self.compute_features(pred_normalized)
            # feat_loss = self.feat_loss(features, feat)
            feat_loss = 0
            spectral_loss = self.feat_weight * feat_loss + self.mrstft_weight * spec_loss
        else:
            spectral_loss = 0
            spec_loss = 0
            feat_loss = 0
        self.logger.experiment.add_scalar("Feature_loss/Train",
                                          feat_loss, global_step=self.global_step)
        self.logger.experiment.add_scalar("MRSTFT_loss/Train",
                                          spec_loss, global_step=self.global_step)
        self.logger.experiment.add_scalar("Total_Spectral_loss/Train",
                                          spectral_loss, global_step=self.global_step)
        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        clean, processed, feat, label, conditioning, fx_class = batch
        batch_size = processed.shape[0]
        pred = self.forward(feat, conditioning)
        # print(pred, label)
        loss = self.loss(pred, label)
        pred_clone = pred.clone()
        # mask predictions according to fx_class
        pred_clone[:, :5] *= (fx_class[:, None] == 0)
        label[:, :5] *= (fx_class[:, None] == 0)
        pred_clone[:, 5:8] *= (fx_class[:, None] == 1)
        label[:, 5:8] *= (fx_class[:, None] == 1)
        pred_clone[:, 8:] *= (fx_class[:, None] == 2)
        label[:, 8:] *= (fx_class[:, None] == 2)
        loss = self.loss(pred_clone, label)
        self.log("loss/test", loss)
        pred_per_fx = [pred[:, :5], pred[:, 5:8], pred[:, 8:]]
        # loss = 0
        self.logger.experiment.add_scalar("Param_loss/Val", loss, global_step=self.global_step)
        scalars = {}
        for (i, val) in enumerate(torch.mean(torch.abs(pred_clone - label), 0)):
            scalars[f'{i}'] = val
        self.logger.experiment.add_scalars("Param_distance/Val", scalars, global_step=self.global_step)
        if self.monitor_spectral_loss:
            pred_per_fx = [ppf.to("cpu") for ppf in pred_per_fx]
            # self.pred = pred
            # self.pred.retain_grad()
            rec = torch.zeros(batch_size, clean.shape[-1], device=self.device)
            for (i, snd) in enumerate(clean):
                snd_norm = snd / torch.max(torch.abs(snd))
                snd_norm = snd_norm + torch.randn_like(snd_norm) * 1e-9
                tmp = self.board_layers[fx_class[i]].forward(snd_norm.cpu(), pred_per_fx[fx_class[i]][i])
                rec[i] = tmp.clone() / torch.max(torch.abs(tmp))
            target_normalized = processed[:, 0, :] / torch.max(torch.abs(processed[:, 0, :]), dim=-1, keepdim=True)[0]
            pred_normalized = rec.to(self.device)
            spec_loss = self.spectral_loss(pred_normalized, target_normalized)
            # features = self.compute_features(pred_normalized)
            # feat_loss = self.feat_loss(features, feat)
            feat_loss = 0
            spectral_loss = self.feat_weight * feat_loss + self.mrstft_weight * spec_loss
        else:
            spectral_loss = 0
            spec_loss = 0
            feat_loss = 0
        self.logger.experiment.add_scalar("Feature_loss/Val",
                                          feat_loss, global_step=self.global_step)
        self.logger.experiment.add_scalar("MRSTFT_loss/Val",
                                          spec_loss, global_step=self.global_step)
        self.logger.experiment.add_scalar("Total_Spectral_loss/Val",
                                          spectral_loss, global_step=self.global_step)
        return loss

    def on_validation_end(self) -> None:
        clean, processed, feat, label, conditioning, fx_class = next(iter(self.trainer.val_dataloaders[0]))
        if conditioning[0] != 'None':
            conditioning = conditioning.to(self.device)
        fx_class = fx_class.to(self.device)
        pred = self.forward(feat.to(self.device), conditioning=conditioning)
        pred = pred.to("cpu")
        pred_per_fx = [pred[:, :5], pred[:, 5:8], pred[:, 8:]]
        rec = torch.zeros(clean.shape[0], clean.shape[-1], device=self.device)  # TODO: fix hardcoded value
        # features = self.compute_features(processed[:, 0, :].to(self.device))
        for (i, snd) in enumerate(clean):
            rec[i] = self.board_layers[fx_class[i]].forward(snd, pred_per_fx[fx_class[i]][i])
        for l in range(self.audiologs):
            self.logger.experiment.add_text(f"Audio/{l}/Original_feat",
                                            str(feat[l]), global_step=self.global_step)
            # self.logger.experiment.add_text(f"Audio/{l}/Predicted_feat",
            #                                 str(features[l]), global_step=self.global_step)
            self.logger.experiment.add_audio(f"Audio/{l}/Original", processed[l] / torch.max(torch.abs(processed[l])),
                                             sample_rate=self.rate, global_step=self.global_step)
            self.logger.experiment.add_audio(f"Audio/{l}/Matched", rec[l] / torch.max(torch.abs(rec[l])),
                                             sample_rate=self.rate, global_step=self.global_step)
            self.logger.experiment.add_text(f"Audio/{l}/Predicted_params", str(pred[l]), global_step=self.global_step)
            if not self.out_of_domain:
                self.logger.experiment.add_text(f"Audio/{l}/Matched_params", str(pred[l]), global_step=self.global_step)
                self.logger.experiment.add_text(f"Audio/{l}/Original_params", str(label[l]),
                                                global_step=self.global_step)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate)
        return optimizer

    def compute_features(self, audio):
        audio = audio + torch.randn_like(audio) * 1e-6
        pitch = Ft.pitch_curve(audio, self.rate, None, None, torch_compat=True)
        phase = Ft.phase_fmax_batch(audio, transform=self.feature_spectro)
        rms = Ft.rms_energy(audio, torch_compat=True)
        pitch_delta = Fc.estim_derivative(pitch, torch_compat=True)
        phase_delta = Fc.estim_derivative(phase, torch_compat=True)
        rms_delta = Fc.estim_derivative(rms, torch_compat=True)
        pitch_fft_max, pitch_freq = Fc.fft_max_batch(pitch,
                                                     num_max=2,
                                                     zero_half_width=32)
        pitch_delta_fft_max, pitch_delta_freq = Fc.fft_max_batch(pitch_delta,
                                                                 num_max=2,
                                                                 zero_half_width=32)
        rms_delta_fft_max, rms_delta_freq = Fc.fft_max_batch(rms_delta,
                                                             num_max=2,
                                                             zero_half_width=32)
        phase_delta_fft_max, phase_delta_freq = Fc.fft_max_batch(phase_delta,
                                                                 num_max=2,
                                                                 zero_half_width=32)
        phase_fft_max, phase_freq = Fc.fft_max_batch(phase, num_max=2, zero_half_width=32)
        rms_fft_max, rms_freq = Fc.fft_max_batch(rms, num_max=2, zero_half_width=32)
        # print("rms_freq: ", rms_freq.requires_grad, rms_freq.grad_fn)
        # print("rms_fft_max: ", rms_fft_max.requires_grad, rms_fft_max.grad_fn)
        rms_std = Fc.f_std(rms, torch_compat=True)
        # print("rms_std: ", rms_std.requires_grad, rms_std.grad_fn)
        rms_skew = Fc.f_skew(rms, torch_compat=True)
        # print("rms_skew: ", rms_skew.requires_grad, rms_skew.grad_fn)
        rms_delta_std = Fc.f_std(rms_delta, torch_compat=True)
        # print("rms_delta_std: ", rms_delta_std.requires_grad, rms_delta_std.grad_fn)
        rms_delta_skew = Fc.f_skew(rms_delta, torch_compat=True)
        # print("rms_delta_skew: ", rms_delta_skew.requires_grad, rms_delta_skew.grad_fn)
        # print(pitch_fft_max)
        # print("pitch_freq: ", pitch_freq[:, 0] / 512)
        # print("pitch_delta_freq: ", pitch_delta_freq[:, 1]/512)
        features = torch.stack((phase_fft_max[:, 0], phase_freq[:, 0] / 512,
                                rms_fft_max[:, 0], rms_freq[:, 0] / 512,
                                phase_fft_max[:, 1], phase_freq[:, 1] / 512,
                                rms_fft_max[:, 1], rms_freq[:, 1] / 512,
                                rms_delta_fft_max[:, 0], rms_delta_freq[:, 0] / 512,
                                rms_delta_fft_max[:, 1], rms_delta_freq[:, 1] / 512,
                                phase_delta_fft_max[:, 0], phase_delta_freq[:, 0] / 512,
                                phase_delta_fft_max[:, 1], phase_delta_freq[:, 1] / 512,
                                pitch_delta_fft_max[:, 0], pitch_delta_freq[:, 0] / 512,
                                pitch_delta_fft_max[:, 1], pitch_delta_freq[:, 1] / 512,
                                pitch_fft_max[:, 0], pitch_freq[:, 0] / 512,
                                pitch_fft_max[:, 1], pitch_freq[:, 1] / 512,
                                rms_std, rms_delta_std, rms_skew, rms_delta_skew
                                ), dim=1)
        onsets, activations = Ft.onset_detection(audio, self.rate, self.filterbank)
        mfccs = self.mfcc_transform(audio)
        mfccs = torch.mean(mfccs, dim=-1)
        features = torch.cat((features, onsets, activations, mfccs), dim=1)
        # print("BEFORE SCALING", features[:, 20])
        out = self.scaler.transform(features)
        # print("OUUUUUUT", out[:, 20])
        return out

if __name__ == "__main__":
    # df = pd.read_csv("/home/alexandre/dataset/modulation_delay_distortion_guitar_mono_cut/data.csv", index_col=0)
    CLEAN_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_dry_22050_cut")
    PROCESSED_PATH = pathlib.Path("/home/alexandre/dataset/modulation_delay_distortion_guitar_mono_cut")
    OUT_OF_DOMAIN_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_modulation_delay_distortion_22050_cut")
    NUM_FEATURES = 211
    HIDDEN_SIZE = 100
    NUM_HIDDEN_LAYERS = 5
    PARAM_RANGE_DISTORTION = [(0, 60),
                              (50, 500), (-10, 10), (0.5, 2),
                              (500, 2000), (-10, 10), (0.5, 2)]
    PARAM_RANGE_DELAY = [(0, 1), (0, 1), (0, 1)]
    PARAM_RANGE_MODULATION = [(0.1, 10), (0, 1), (0, 20), (0, 1), (0, 1)]
    data = pd.read_csv(PROCESSED_PATH / 'data_full.csv', index_col=0)
    sss_in = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=2)
    y_in = data["fx_class"]
    X_in = data.iloc[:, :-1]
    train_index, val_index = next(iter(sss_in.split(X_in, y_in)))
    in_train = X_in.iloc[train_index]
    # scaler = sklearn.preprocessing.StandardScaler()
    # FEAT_COL = data.columns.str.startswith('f-')
    # FEAT_COL = data.columns[FEAT_COL]
    # print(FEAT_COL)
    # scaler.fit(in_train[FEAT_COL])
    # print(scaler.mean_)
    with open("/home/alexandre/logs/SimpleMLP11aout/211feat-conditioning-10_hidden/scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)


    datamodule = FeaturesDataModule(CLEAN_PATH, PROCESSED_PATH, OUT_OF_DOMAIN_PATH,
                                    in_scaler_mean=scaler.mean_, in_scaler_std=np.sqrt(scaler.var_),
                                    out_scaler_mean=scaler.mean_, out_scaler_std=np.sqrt(scaler.var_),
                                    seed=2, batch_size=64, fx_feat=True, clf_feat=True,
                                    conditioning=True, classes2keep=[0, 1, 2], csv_name='data_full.csv')
    # datamodule.setup()
    SAVE_PATH = pathlib.Path("/tmp")
    if not SAVE_PATH.exists():
        os.mkdir(SAVE_PATH)
    with open(SAVE_PATH / "scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    mlp = SimpleMLP(NUM_FEATURES, HIDDEN_SIZE, NUM_HIDDEN_LAYERS,
                    scaler.mean_, np.sqrt(scaler.var_), PARAM_RANGE_MODULATION,
                    PARAM_RANGE_DELAY, PARAM_RANGE_DISTORTION, tracker=True,
                    monitor_spectral_loss=False)
    checkpoint_callback = ModelCheckpoint(save_top_k=3, monitor='loss/test')
    early_stopping = EarlyStopping(monitor='loss/test', patience=3)
    logger = TensorBoardLogger("/home/alexandre/logs/SimpleMLP11aout", name="163feat-conditioning-10_hidden")
    trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=1000, log_every_n_steps=100,
                         callbacks=[checkpoint_callback, early_stopping])

    trainer.fit(mlp, datamodule=datamodule)