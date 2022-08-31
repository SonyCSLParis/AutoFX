"""
ResNet network for AutoFX with conditional features added to the last layer
"""
import math
from typing import List, Tuple, Any, Optional

import auraloss
import pytorch_lightning as pl
import torch
import torchaudio
from carbontracker.tracker import CarbonTracker
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
import pedalboard as pdb

from source.models.film_layer import FilmLayer
from source.data.datasets import TorchStandardScaler
from source.models.mbfx_layer import MBFxLayer
from source.multiband_fx import MultiBandFX
from source.models.custom_distortion import CustomDistortion
from source.models.resnet_layers import ResNet
import data.functional as Fc
import data.features as Ft
import torchaudio
import source.util as util
from data import superflux


class CAFx(pl.LightningModule):
    def compute_features(self, audio):
        audio = audio + torch.randn_like(audio) * 1e-4
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
        features = torch.cat((features, onsets, activations), dim=1)
        # print("BEFORE SCALING", features[:, 20])
        out = self.scaler.transform(features)
        # print("OUUUUUUT", out[:, 20])
        return out

    def __init__(self, fx: str, num_bands: int, param_range: list,
                 cond_feat: int, scaler_mean: list, scaler_std: list,
                 tracker: bool = False,
                 rate: int = 22050, total_num_bands: int = None,
                 fft_size: int = 1024, hop_size: int = 256, audiologs: int = 4, loss_weights: list[float] = [1, 1],
                 mrstft_fft: list[int] = [64, 128, 256, 512, 1024, 2048],
                 mrstft_hop: list[int] = [16, 32, 64, 128, 256, 512],
                 learning_rate: float = 0.0001, out_of_domain: bool = False,
                 spectro_power: int = 2, mel_spectro: bool = True, mel_num_bands: int = 128,
                 penalty_1: float = 0, penalty_0: float = 0, feat_weight: float = 0.5, mrstft_weight: float = 0.5,
                 loss_stamps: list = None, freeze_layers: list = None,
                 reverb: bool = False, with_film: bool = False, disable_features: bool=False,
                 monitor_spectral_loss: bool = False):
        super().__init__()
        if isinstance(fx, str) and fx != "distortion":
            fx = [util.str2pdb(fx)]
        if fx == "distortion":
            mbfx = CustomDistortion()
        if total_num_bands is None:
            total_num_bands = num_bands
        self.total_num_bands = total_num_bands
        if fx == "distortion":
            self.mbfx = mbfx
        else:
            self.mbfx = MultiBandFX(fx, total_num_bands, device=torch.device('cpu'))
        self.num_params = num_bands * self.mbfx.total_num_params_per_band
        self.reverb = reverb
        self.scaler = TorchStandardScaler()
        # self.scaler.mean = torch.tensor(scaler_mean, device=self.device)
        # self.scaler.std = torch.tensor(scaler_std, device=self.device)
        self.scaler.mean = torch.tensor(scaler_mean, device=torch.device('cuda'))
        self.scaler.std = torch.tensor(scaler_std, device=torch.device('cuda'))
        filt = superflux.Filter(2048 // 2 + 1, rate=22050, bands=24, fmin=30, fmax=17000, equal=False)
        self.filterbank = filt.filterbank
        # self.filterbank = self.filterbank.to(self.device)
        self.filterbank = self.filterbank.to(torch.device("cuda"))
        print(self.scaler.mean)
        print(self.scaler.std)
        if reverb:
            self.num_params -= 1
        self.resnet = ResNet(self.num_params, end_with_fcl=False, num_channels=64, with_film=with_film)
        self.with_film = with_film
        if self.with_film:
            self.film1_1 = FilmLayer(1, 64)
            self.film1_2 = FilmLayer(1, 64)
            self.film2_1 = FilmLayer(1, 128)
            self.film2_2 = FilmLayer(1, 128)
            self.film3_1 = FilmLayer(1, 256)
            self.film3_2 = FilmLayer(1, 256)
            nn.init.normal_(self.film1_1.linear1.weight, mean=1, std=0.1)
            nn.init.normal_(self.film1_1.linear2.weight, mean=0, std=0.1)
            nn.init.normal_(self.film1_2.linear1.weight, mean=1, std=0.1)
            nn.init.normal_(self.film1_2.linear2.weight, mean=0, std=0.1)
            nn.init.normal_(self.film2_1.linear1.weight, mean=1, std=0.1)
            nn.init.normal_(self.film2_1.linear2.weight, mean=0, std=0.1)
            nn.init.normal_(self.film2_2.linear1.weight, mean=1, std=0.1)
            nn.init.normal_(self.film2_2.linear2.weight, mean=0, std=0.1)
            nn.init.normal_(self.film3_1.linear1.weight, mean=1, std=0.1)
            nn.init.normal_(self.film3_1.linear2.weight, mean=0, std=0.1)
            nn.init.normal_(self.film3_2.linear1.weight, mean=1, std=0.1)
            nn.init.normal_(self.film3_2.linear2.weight, mean=0, std=0.1)
        self.cond_feat = cond_feat
        # TODO: Make this cleaner
        if reverb:
            fcl_size = 4096
        else:
            fcl_size = fft_size * 256 // hop_size
        self.disable_features = disable_features
        if disable_features:
            cond_feat = 0
        self.fcl1 = nn.Linear(fcl_size + cond_feat, fcl_size // 2)
        self.fcl2 = nn.Linear(fcl_size // 2, self.num_params)
        # self.fcl = nn.Linear(fcl_size + cond_feat, self.num_params)
        # nn.init.xavier_normal_(self.fcl.weight, gain=math.sqrt(2))
        # nn.init.xavier_normal_(self.fcl1.weight, gain=math.sqrt(2))
        # nn.init.xavier_normal_(self.fcl2.weight, gain=math.sqrt(2))
        self.relu = nn.ReLU(inplace=True)
        self.activation = nn.Sigmoid()
        self.learning_rate = learning_rate
        self.loss = nn.MSELoss()
        self.feat_loss = nn.MSELoss()
        self.tracker_flag = tracker
        self.tracker = None
        self.mrstft = auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=mrstft_fft,
                                                            hop_sizes=mrstft_hop,
                                                            win_lengths=mrstft_fft,
                                                            w_phs=1,
                                                            w_sc=0)
        self.spectral_loss = self.mrstft
        self.num_bands = num_bands
        if isinstance(param_range[0], str):
            param_range = util.param_range_from_cli(param_range)
        self.param_range = param_range
        self.rate = rate
        self.feature_spectro = torchaudio.transforms.Spectrogram(n_fft=2048, hop_length=256, power=None)
        self.out_of_domain = out_of_domain
        if loss_stamps is None:
            self.loss_weights = [1, 0]
        else:
            self.loss_weights = loss_weights
        if mel_spectro:
            self.spectro = torchaudio.transforms.MelSpectrogram(n_fft=fft_size, hop_length=hop_size,
                                                                sample_rate=self.rate,
                                                                power=spectro_power, n_mels=mel_num_bands)
        else:
            self.spectro = torchaudio.transforms.Spectrogram(n_fft=fft_size, hop_length=hop_size, power=spectro_power)
        self.mbfx_layer = MBFxLayer(self.mbfx, self.rate, self.param_range, fake_num_bands=self.num_bands)
        self.inv_spectro = torchaudio.transforms.InverseSpectrogram(n_fft=fft_size, hop_length=hop_size)
        self.audiologs = audiologs
        self.tmp = None
        self.loss_stamps = loss_stamps
        self.num_steps_per_epoch = None
        self.num_steps_per_train = None
        self.num_steps_per_valid = None
        self.penalty_1 = penalty_1
        self.penalty_0 = penalty_0
        self.feat_weight = feat_weight
        self.mrstft_weight = mrstft_weight
        self.freeze_layers = freeze_layers
        if freeze_layers is not None:
            for f in freeze_layers:
                self.resnet.freeze(f)
        self.monitor_spectral_loss = monitor_spectral_loss
        self.save_hyperparameters()

    def forward(self, x, feat, conditioning=None, *args, **kwargs) -> Any:
        if self.with_film:
            # conditioning = conditioning[None, 0]
            conditioning = conditioning[:, None]
            alphas = []
            betas = []
            alpha, beta = self.film1_1(conditioning)
            alphas.append(alpha)
            betas.append(beta)
            alpha, beta = self.film1_2(conditioning)
            alphas.append(alpha)
            betas.append(beta)
            alpha, beta = self.film2_1(conditioning)
            alphas.append(alpha)
            betas.append(beta)
            alpha, beta = self.film2_2(conditioning)
            alphas.append(alpha)
            betas.append(beta)
            alpha, beta = self.film3_1(conditioning)
            alphas.append(alpha)
            betas.append(beta)
            alpha, beta = self.film3_2(conditioning)
            alphas.append(alpha)
            betas.append(beta)
        else:
            alphas, betas = None, None
        out = self.spectro(x)
        out = self.resnet(out, alphas, betas)
        if not self.disable_features:
            out = torch.cat((out, feat), dim=-1)
        out = self.fcl1(out)
        # print("before:", out)
        out = self.relu(out)
        out = self.fcl2(out)
        ## out = self.fcl(out)
        out = self.activation(out)
        # print("after: ", out)
        if self.reverb:
            # freeze_mode param is always zero
            out = torch.hstack([out, torch.zeros(out.shape[0], 1, device=out.device)])
        return out

    def training_step(self, batch, batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        num_steps_per_epoch = len(self.trainer.train_dataloader) / self.trainer.accumulate_grad_batches
        num_steps_per_epoch = num_steps_per_epoch * self.trainer.num_devices
        if not self.out_of_domain:
            if self.with_film:
                clean, processed, feat, label, conditioning = batch
            else:
                clean, processed, feat, label = batch
                conditioning = None
        else:
            if self.with_film:
                clean, processed, feat, conditioning = batch
            else:
                clean, processed, feat = batch
                conditioning = None
        batch_size = processed.shape[0]
        if self.disable_features:
            pred = self.forward(processed, None, conditioning=conditioning)
        else:
            pred = self.forward(processed, feat, conditioning=conditioning)
        penalty_0 = torch.mean(-1 * torch.log10(pred*0.99))
        penalty_1 = torch.mean(-1 * torch.log10(1 - 0.99*pred))
        if not self.out_of_domain:
            loss = self.loss(pred, label)
            # loss = 0
            self.logger.experiment.add_scalar("Param_loss/Train", loss, global_step=self.global_step)
            scalars = {}
            for (i, val) in enumerate(torch.mean(torch.abs(pred - label), 0)):
                scalars[f'{i}'] = val
            self.logger.experiment.add_scalars("Param_distance/Train", scalars, global_step=self.global_step)
            if self.loss_stamps is not None:
                # TODO: could be moved to optimizers
                if self.trainer.current_epoch < self.loss_stamps[0]:
                    self.loss_weights = [1, 0]
                elif self.trainer.current_epoch < self.loss_stamps[1]:
                    weight = (self.trainer.global_step - (self.loss_stamps[0] * num_steps_per_epoch)) \
                             / ((self.loss_stamps[1] - self.loss_stamps[0]) * num_steps_per_epoch)
                    self.loss_weights = [1 - weight, weight]
                elif self.trainer.current_epoch >= self.loss_stamps[1]:
                    self.loss_weights = [0, 1]
        if self.loss_weights[1] != 0 or self.out_of_domain or self.monitor_spectral_loss:
            pred = pred.to("cpu")
            self.pred = pred
            self.pred.retain_grad()
            rec = torch.zeros(batch_size, clean.shape[-1], device=self.device)
            for (i, snd) in enumerate(clean):
                snd_norm = snd / torch.max(torch.abs(snd))
                snd_norm = snd_norm + torch.randn_like(snd_norm) * 1e-9
                tmp = self.mbfx_layer.forward(snd_norm.cpu(), pred[i])
                rec[i] = tmp.clone()
            target_normalized, pred_normalized = processed[:, 0, :] / torch.max(torch.abs(processed)), rec / torch.max(
                torch.abs(rec))
            spec_loss = self.spectral_loss(pred_normalized, target_normalized)
            features = self.compute_features(pred_normalized)
            feat_loss = self.feat_loss(features, feat)
            spectral_loss = self.feat_weight*feat_loss + self.mrstft_weight*spec_loss
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
        if not self.out_of_domain:
            if self.monitor_spectral_loss:
                # set spectral_loss to 0 to block training on it
                spectral_loss = 0
            total_loss = 100 * loss * self.loss_weights[0] + spectral_loss * self.loss_weights[1]
        else:
            total_loss = spectral_loss + self.penalty_0*penalty_0 + self.penalty_1*penalty_1
        self.logger.experiment.add_scalar("Total_loss/Train", total_loss, global_step=self.global_step)
        # print("MAYBE", target_normalized==pred_normalized)
        return total_loss

    def on_after_backward(self) -> None:
        pass

    def validation_step(self, batch, batch_idx, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        if not self.out_of_domain:
            if self.with_film:
                clean, processed, feat, label, conditioning = batch
            else:
                clean, processed, feat, label = batch
                conditioning = None
        else:
            if self.with_film:
                clean, processed, feat, conditioning = batch
            else:
                clean, processed, feat = batch
                conditioning = None
        # clean = clean.to("cpu")
        batch_size = processed.shape[0]
        if self.disable_features:
            pred = self.forward(processed, None, conditioning=conditioning)
        else:
            pred = self.forward(processed, feat, conditioning=conditioning)
        if not self.out_of_domain:
            loss = self.loss(pred, label)
            self.logger.experiment.add_scalar("Param_loss/test", loss, global_step=self.global_step)
            scalars = {}
            for (i, val) in enumerate(torch.mean(torch.abs(pred - label), 0)):
                scalars[f'{i}'] = val
            self.logger.experiment.add_scalars("Param_distance/test", scalars, global_step=self.global_step)
        if self.loss_weights[1] != 0 or self.out_of_domain or self.monitor_spectral_loss:
            pred = pred.to("cpu")
            rec = torch.zeros(batch_size, clean.shape[-1], device=self.device)
            for (i, snd) in enumerate(clean):
                rec[i] = self.mbfx_layer.forward(snd.cpu(), pred[i])
                # target_normalized, pred_normalized = processed[:, 0, :] / torch.max(torch.abs(processed)), rec / torch.max(
                #    torch.abs(rec))
            pred_normalized = rec / torch.max(torch.abs(rec), dim=-1, keepdim=True)[0]
            spec_loss = self.spectral_loss(pred_normalized, processed[:, 0, :])
            # features = self.compute_features(pred_normalized)
            # feat_loss = self.loss(features, feat)
            feat_loss = 0
            spectral_loss = (spec_loss + 1 * feat_loss) / 2
        else:
            spectral_loss = 0
            feat_loss = 0
            spec_loss = 0
        self.logger.experiment.add_scalar("Feature_loss/test",
                                          feat_loss, global_step=self.global_step)
        self.logger.experiment.add_scalar("MRSTFT_loss/test",
                                          spec_loss, global_step=self.global_step)
        self.logger.experiment.add_scalar("Total_Spectral_loss/test",
                                          spectral_loss, global_step=self.global_step)
        if not self.out_of_domain:
            if self.monitor_spectral_loss:
                spectral_loss = 0
            total_loss = 100 * loss * self.loss_weights[0] + spectral_loss * self.loss_weights[1]
        else:
            total_loss = spectral_loss
        self.logger.experiment.add_scalar("Total_loss/test", total_loss, global_step=self.global_step)
        return total_loss

    def on_validation_end(self) -> None:
        if not self.out_of_domain:
            if self.with_film:
                clean, processed, feat, label, conditioning = next(iter(self.trainer.val_dataloaders[0]))
                conditioning = conditioning.to(self.device)
            else:
                clean, processed, feat, label = next(iter(self.trainer.val_dataloaders[0]))
                conditioning = None
        else:
            clean, processed, feat = next(iter(self.trainer.val_dataloaders[0]))
        pred = self.forward(processed.to(self.device), feat.to(self.device), conditioning=conditioning)
        pred = pred.to("cpu")
        rec = torch.zeros(clean.shape[0], clean.shape[-1], device=self.device)  # TODO: fix hardcoded value
        # features = self.compute_features(processed[:, 0, :].to(self.device))
        for (i, snd) in enumerate(clean):
            rec[i] = self.mbfx_layer.forward(snd, pred[i])
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

    def on_train_epoch_start(self) -> None:
        if self.tracker_flag and self.tracker is None:
            self.tracker = CarbonTracker(epochs=self.trainer.max_epochs, epochs_before_pred=10, monitor_epochs=10,
                                         log_dir=self.logger.log_dir, verbose=2)  # TODO: Remove hardcoded values
        if self.tracker_flag:
            self.tracker.epoch_start()

    def on_train_epoch_end(self) -> None:
        if self.tracker_flag:
            self.tracker.epoch_end()

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)  # TODO: Remove hardcoded values
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        # lr_schedulers = {"scheduler": scheduler, "interval": "epoch"}
        # return {"optimizer": optimizer, "lr_scheduler": lr_schedulers}
        return optimizer
