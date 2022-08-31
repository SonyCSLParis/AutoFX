"""
Convolutional Neural Network for parameter estimation
"""

import os
import pathlib
import pedalboard as pdb
from typing import Any, Optional, Tuple, List

from torch.autograd import Variable
import torchaudio.transforms
from carbontracker.tracker import CarbonTracker
import auraloss
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import sys
from source.multiband_fx import MultiBandFX
from math import floor
import math
from source.models.mbfx_layer import MBFxLayer
from source.models.resnet_layers import ResNet
sys.path.append('../..')
from source.losses import PitchLoss

def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.softplus(-2. * x) - math.log(2.0)
    return torch.mean(_log_cosh(y_pred - y_true))

class LogCoshLoss(torch.nn.Module):
    """
    From:
    https://datascience.stackexchange.com/questions/96271/logcoshloss-on-pytorch
    """
    def __init__(self):
        super().__init__()

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        return log_cosh_loss(y_pred, y_true)


class AutoFX(pl.LightningModule):
    def _shape_after_conv(self, x: torch.Tensor or Tuple):
        """
        Return shape after Conv2D according to https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        :param x:
        :return:
        """
        if isinstance(x, torch.Tensor):
            batch_size, c_in, h_in, w_in = x.shape
        else:
            batch_size, c_in, h_in, w_in = x
        for seq in self.conv:
            conv = seq[0]
            h_out = floor(
                (h_in + 2 * conv.padding[0] - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1) / conv.stride[0] + 1)
            w_out = floor(
                (w_in + 2 * conv.padding[1] - conv.dilation[1] * (conv.kernel_size[1] - 1) - 1) / conv.stride[1] + 1)
            h_in = h_out
            w_in = w_out
        c_out = self.conv[-1][0].out_channels
        return batch_size, c_out, h_out, w_out

    def __init__(self, fx: pdb.Plugin, num_bands: int, param_range: List[Tuple], tracker: bool = False,
                 rate: int = 22050, file_size: int = 44100, total_num_bands: int = None,
                 conv_ch: list[int] = [128, 128, 64, 64, 64, 64, 64], conv_k: list[int] = [3, 3, 5, 5, 5, 7, 7],
                 conv_stride: list[int] = [2, 2, 2, 2, 2, 1, 1],
                 fft_size: int = 1024, hop_size: int = 256, audiologs: int = 4, loss_weights: list[float] = [1, 1],
                 mrstft_fft: list[int] = [64, 128, 256, 512, 1024, 2048],
                 mrstft_hop: list[int] = [16, 32, 64, 128, 256, 512],
                 learning_rate: float = 0.001, out_of_domain: bool = False,
                 spectro_power: int = 2, mel_spectro: bool = True, mel_num_bands: int = 128,
                 loss_stamps: list = None,
                 device=torch.device('cuda')):  # TODO: change
        super().__init__()
        if total_num_bands is None:
            total_num_bands = num_bands
        self.total_num_bands = total_num_bands
        self.mbfx = MultiBandFX(fx, total_num_bands, device=torch.device('cpu'))
        self.num_params = num_bands * self.mbfx.total_num_params_per_band
        self.resnet = ResNet(self.num_params)
        self.activation = nn.Sigmoid()
        self.learning_rate = learning_rate
        # self.loss = nn.L1Loss()
        self.loss = nn.MSELoss()
        self.tracker_flag = tracker
        self.tracker = None
        self.mrstft = auraloss.freq.MultiResolutionSTFTLoss(mrstft_fft,
                                                            mrstft_hop,
                                                            mrstft_fft,
                                                            w_log_mag=1,
                                                            w_lin_mag=1,
                                                            w_phs=1,
                                                            sample_rate=rate,
                                                            device=device)  # TODO: Manage device properly
        # self.spectral_loss = auraloss.time.ESRLoss()
        # self.spectral_loss = self.mrstft
        self.spectral_loss = PitchLoss(rate=rate)
        # for stft_loss in self.mrstft.stft_losses:
        #    stft_loss = stft_loss.cuda()
        self.num_bands = num_bands
        self.param_range = param_range
        self.rate = rate
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

    def forward(self, x, *args, **kwargs) -> Any:
        out = self.spectro(x)
        out = self.resnet(out)
        # print("before:", out)
        out = self.activation(out)
        # print("after: ", out)
        return out

    def training_step(self, batch, batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        num_steps_per_epoch = len(self.trainer.train_dataloader) / self.trainer.accumulate_grad_batches
        if not self.out_of_domain:
            clean, processed, label = batch
        else:
            clean, processed = batch
        batch_size = processed.shape[0]
        pred = self.forward(processed)
        if not self.out_of_domain:
            loss = self.loss(pred, label)
            self.logger.experiment.add_scalar("Param_loss/Train", loss, global_step=self.global_step)
            scalars = {}
            for (i, val) in enumerate(torch.mean(torch.abs(pred - label), 0)):
                scalars[f'{i}'] = val
            self.logger.experiment.add_scalars("Param_distance/Train", scalars, global_step=self.global_step)
            if self.loss_stamps is not None:
                # TODO: could be moved to optimizers
                if self.trainer.current_epoch < self.loss_stamps[0]:  # TODO: make it cleaner
                    self.loss_weights = [1, 0]
                elif self.trainer.current_epoch < self.loss_stamps[1]:
                    weight = (self.trainer.global_step - (self.loss_stamps[0] * num_steps_per_epoch))\
                             / ((self.loss_stamps[1] - self.loss_stamps[0]) * num_steps_per_epoch)
                    self.loss_weights = [1 - weight, weight]
                elif self.trainer.current_epoch >= self.loss_stamps[1]:
                    self.loss_weights = [0, 1]
        pred = pred.to("cpu")
        rec = torch.zeros(batch_size, clean.shape[-1], device=self.device)
        for (i, snd) in enumerate(clean):
            # self.mbfx_layer.params = pred[i]
            tmp = self.mbfx_layer.forward(snd.cpu(), pred[i])
            rec[i] = tmp
        self.tmp = rec
        if self.loss_weights[1] != 0 or self.out_of_domain:
            # target_aligned, pred_aligned = time_align_signals(processed[:, 0, :], rec)
            target_aligned, pred_aligned = processed[:, 0, :].clone(), rec.clone()
            spec_loss = self.spectral_loss(pred_aligned, target_aligned)
            time_loss = 0
            # time_loss = self.time_loss(pred_aligned, target_aligned, aligned=True)
            spectral_loss = spec_loss + 1000 * time_loss
        else:
            spectral_loss = 0
            spec_loss = 0
            time_loss = 0
        self.logger.experiment.add_scalar("Time_loss/Train",
                                          time_loss, global_step=self.global_step)
        self.logger.experiment.add_scalar("MRSTFT_loss/Train",
                                          spec_loss, global_step=self.global_step)
        self.logger.experiment.add_scalar("Total_Spectral_loss/Train",
                                          spectral_loss, global_step=self.global_step)
        if not self.out_of_domain:
            total_loss = 100 * loss * self.loss_weights[0] + spectral_loss * self.loss_weights[1]
        else:
            total_loss = spectral_loss
        self.logger.experiment.add_scalar("Total_loss/Train", total_loss, global_step=self.global_step)
        return total_loss

    def on_before_backward(self, loss: torch.Tensor) -> None:
        # self.tmp.backward(torch.ones_like(self.tmp))
        pass

    def on_after_backward(self) -> None:
        # print(self.mbfx_layer.params.grad_fn)
        if self.out_of_domain or self.loss_weights[1] != 0 or True:
            # pass
            # self.logger.experiment.add_histogram("Grad/params", self.mbfx_layer.params.grad, global_step=self.global_step)
            self.logger.experiment.add_histogram("Weights/resnet_fcl", self.resnet.fcl.weight, global_step=self.global_step)
            self.logger.experiment.add_histogram("Grad/resnet_fcl", self.resnet.fcl.weight.grad, global_step=self.global_step)
            self.logger.experiment.add_histogram("Weights/resnet_conv1", self.resnet.conv1[0].weight, global_step=self.global_step)
            self.logger.experiment.add_histogram("Grad/resnet_conv1", self.resnet.conv1[0].weight.grad,
                                                 global_step=self.global_step)

    def validation_step(self, batch, batch_idx, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        if not self.out_of_domain:
            clean, processed, label = batch
        else:
            clean, processed = batch
        clean = clean.to("cpu")
        batch_size = processed.shape[0]
        pred = self.forward(processed)
        if not self.out_of_domain:
            loss = self.loss(pred, label)
            self.logger.experiment.add_scalar("Param_loss/test", loss, global_step=self.global_step)
            scalars = {}
            for (i, val) in enumerate(torch.mean(torch.abs(pred - label), 0)):
                scalars[f'{i}'] = val
            self.logger.experiment.add_scalars("Param_distance/test", scalars, global_step=self.global_step)
        pred = pred.to("cpu")
        rec = torch.zeros(batch_size, clean.shape[-1], device=self.device)  # TODO: fix hardcoded value
        for (i, snd) in enumerate(clean):
            # self.mbfx_layer.params = nn.Parameter(pred[i])
            rec[i] = self.mbfx_layer.forward(snd, pred[i])
        if self.loss_weights[1] != 0 or self.out_of_domain:
            # target_aligned, pred_aligned = time_align_signals(processed[:, 0, :], rec)
            target_aligned, pred_aligned = processed[:, 0, :].clone(), rec.clone()
            spec_loss = self.spectral_loss(pred_aligned, target_aligned)
            time_loss = 0
            # time_loss = self.time_loss(pred_aligned, target_aligned, aligned=True)
            spectral_loss = spec_loss + 1000 * time_loss
        else:
            spectral_loss = 0
            spec_loss = 0
            time_loss = 0
        self.logger.experiment.add_scalar("Time_loss/test",
                                          time_loss, global_step=self.global_step)
        self.logger.experiment.add_scalar("MRSTFT_loss/test",
                                          spec_loss, global_step=self.global_step)
        self.logger.experiment.add_scalar("Total_Spectral_loss/test",
                                          spectral_loss, global_step=self.global_step)
        if not self.out_of_domain:
            total_loss = 100 * loss * self.loss_weights[0] + spectral_loss * self.loss_weights[1]
        else:
            total_loss = spectral_loss
        self.logger.experiment.add_scalar("Total_loss/test", total_loss, global_step=self.global_step)
        return total_loss

    def on_validation_end(self) -> None:
        if not self.out_of_domain:
            clean, processed, label = next(iter(self.trainer.val_dataloaders[0]))
        else:
            clean, processed = next(iter(self.trainer.val_dataloaders[0]))
        pred = self.forward(processed.to(self.device))
        pred = pred.to("cpu")
        rec = torch.zeros(clean.shape[0], clean.shape[-1], device=self.device)  # TODO: fix hardcoded value
        for (i, snd) in enumerate(clean):
            # self.mbfx_layer.params = nn.Parameter(pred[i])
            rec[i] = self.mbfx_layer.forward(snd, pred[i])
        for l in range(self.audiologs):
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)  # TODO: Remove hardcoded values
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        lr_schedulers = {"scheduler": scheduler, "interval": "epoch"}
        return {"optimizer": optimizer, "lr_scheduler": lr_schedulers}
