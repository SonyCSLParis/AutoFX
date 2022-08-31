"""
Convolutional Neural Network for parameter estimation
"""

import os
import pathlib
import pedalboard as pdb
from typing import Any, Optional, Tuple

import torchaudio.transforms
from carbontracker.tracker import CarbonTracker
import auraloss
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
import pytorch_lightning as pl
import sys
from multiband_fx import MultiBandFX
from math import floor
sys.path.append('../..')


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
            h_out = floor((h_in + 2*conv.padding[0] - conv.dilation[0]*(conv.kernel_size[0] - 1) - 1)/conv.stride[0] + 1)
            w_out = floor((w_in + 2*conv.padding[1] - conv.dilation[1]*(conv.kernel_size[1] - 1) - 1)/conv.stride[1] + 1)
            h_in = h_out
            w_in = w_out
        c_out = self.conv[-1][0].out_channels
        return batch_size, c_out, h_out, w_out

    def __init__(self, fx: pdb.Plugin, num_bands: int, tracker: bool = False, rate: int = 22050,
                 conv_ch: list[int] = [64, 64, 64], conv_k: list[int] = [5, 5, 5],
                 fft_size: int = 1024, hop_size: int = 256, audiologs: int = 4,
                 mrstft_fft: list[int] = [64, 128, 256, 512, 1024, 2048],
                 mrstft_hop: list[int] = [16, 32, 64, 128, 256, 512],
                 spectro_power: int = 2, hidden_size: int = 512):
        super().__init__()
        self.conv = nn.ModuleList([])
        self.mbfx = MultiBandFX(fx, num_bands)
        self.num_params = num_bands * len(self.mbfx.settings[0])
        for c in range(len(conv_ch)):
            if c == 0:
                self.conv.append(nn.Sequential(nn.Conv2d(1, conv_ch[c], conv_k[c]), nn.BatchNorm2d(conv_ch[c]), nn.ReLU()))
            else:
                self.conv.append(nn.Sequential(nn.Conv2d(conv_ch[c-1], conv_ch[c], conv_k[c]), nn.BatchNorm2d(conv_ch[c]), nn.ReLU()))
        _, _, h_out, _ = self._shape_after_conv((None, None, fft_size//2 + 1, 0))
        self.gru = nn.GRU(h_out * conv_ch[-1], hidden_size, batch_first=True)
        self.hidden_size = hidden_size
        self.fcl = nn.Linear(hidden_size, self.num_params)
        self.activation = nn.Sigmoid()
        self.loss = nn.L1Loss()
        self.tracker_flag = tracker
        self.tracker = None
        self.mrstft = auraloss.freq.MultiResolutionSTFTLoss(mrstft_fft,
                                                            mrstft_hop,
                                                            mrstft_fft,
                                                            device="cpu")       # TODO: Manage device properly
        self.num_bands = num_bands
        self.rate = rate
        self.spectro = torchaudio.transforms.Spectrogram(n_fft=fft_size, hop_length=hop_size, power=spectro_power)
        self.inv_spectro = torchaudio.transforms.InverseSpectrogram(n_fft=fft_size, hop_length=hop_size)
        self.audiologs = audiologs

    def forward(self, x, *args, **kwargs) -> Any:
        x = self.spectro(x)
        for conv in self.conv:
            x = conv(x)
        batch_size, channels, h_out, w_out = x.shape
        x = x.view(batch_size, w_out, channels, h_out)
        x = x.view(batch_size, w_out, -1)
        x, _ = self.gru(x, torch.zeros(1, batch_size, self.hidden_size, device=x.device))
        x = self.fcl(x)
        x = torch.mean(x, 1)
        x = self.activation(x)
        return x

    def training_step(self, batch, batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        clean, processed, label = batch
        batch_size = processed.shape[0]
        pred = self.forward(processed)
        rec = torch.zeros(batch_size, clean.shape[-1] - 1)        # TODO: Remove hardcoded values
        for (i, snd) in enumerate(clean):
            for b in range(self.num_bands):
                # TODO: Make it fx agnostic
                self.mbfx.mbfx[b][0].drive_db = pred[i][b] * 50 + 10                        # TODO: how to remove hardcoded values?
                self.mbfx.mbfx[b][1].gain_db = (pred[i][self.num_bands + b] - 0.5) * 20
            rec[i] = self.mbfx(snd.to(torch.device('cpu')), self.rate)
        self.log("Spectral_loss/Train",
                 self.mrstft(rec.to(torch.device('cpu')), processed.to(torch.device('cpu'))))  # TODO: fix device management
        loss = self.loss(pred, label)
        self.log("Total_loss/train", loss)
        scalars = {}
        for (i, val) in enumerate(torch.mean(torch.abs(pred - label), 0)):
            scalars[f'{i}'] = val
        self.logger.experiment.add_scalars("Param_distance/Train", scalars, global_step=self.global_step)
        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        clean, processed, label = batch
        batch_size = processed.shape[0]
        pred = self.forward(processed)
        rec = torch.zeros(batch_size, clean.shape[-1] - 1)  # TODO: fix hardcoded value
        for (i, snd) in enumerate(clean):
            for b in range(self.num_bands):
                # TODO: Make it fx agnostic
                self.mbfx.mbfx[b][0].drive_db = pred[i][b] * 50 + 10                    # TODO: How to remove hardcoded values?
                self.mbfx.mbfx[b][1].gain_db = (pred[i][self.num_bands + b] - 0.5) * 20
            rec[i] = self.mbfx(snd.to(torch.device('cpu')),                             # TODO: Could MBFX be applied on GPU?
                               self.rate)
        self.log("Spectral_loss/test",
                 self.mrstft(rec.to(torch.device("cpu")), processed.to(torch.device("cpu"))))  # TODO: Fix device management
        loss = self.loss(pred, label)
        self.log("Total_loss/test", loss)
        for l in range(self.audiologs):
            self.logger.experiment.add_audio(f"Audio/{l}/Original", processed[l],
                                             sample_rate=self.rate, global_step=self.global_step)
            self.logger.experiment.add_text(f"Audio/{l}/Original_params", str(label[l]), global_step=self.global_step)
            self.logger.experiment.add_audio(f"Audio/{l}/Matched", rec[l],
                                             sample_rate=self.rate, global_step=self.global_step)
            self.logger.experiment.add_text(f"Audio/{l}/Matched_params", str(pred[l]), global_step=self.global_step)

        scalars = {}
        for (i, val) in enumerate(torch.mean(torch.abs(pred - label), 0)):
            scalars[f'{i}'] = val
        self.logger.experiment.add_scalars("Param_distance/test", scalars, global_step=self.global_step)
        return loss

    def on_train_epoch_start(self) -> None:
        if self.tracker_flag and self.tracker is None:
            self.tracker = CarbonTracker(epochs=400, epochs_before_pred=10, monitor_epochs=10,
                                         log_dir=self.logger.log_dir, verbose=2)            # TODO: Remove hardcoded values
        if self.tracker_flag:
            self.tracker.epoch_start()

    def on_train_epoch_end(self) -> None:
        if self.tracker_flag:
            self.tracker.epoch_end()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)           # TODO: Remove hardcoded values
        return optimizer
