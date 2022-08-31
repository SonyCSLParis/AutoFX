from typing import Any, Optional

import pedalboard
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from mbfx_layer import MBFxLayer
from multiband_fx import MultiBandFX


class LightNetwork(pl.LightningModule):
    def __init__(self, input_features: int, hidden_layer_size: int,
                 output_size: int, fx: pedalboard.Plugin, param_range, rate: int = 22050,
                 learning_rate: float = 0.001, audiologs: int = 6, out_of_domain: bool = False):
        super(LightNetwork, self).__init__()
        self.in_feat = input_features
        self.hidden_size = hidden_layer_size
        self.out_size = output_size
        self.linear1 = nn.Linear(self.in_feat, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = nn.Linear(self.hidden_size, self.out_size)
        self.activation = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(self.in_feat)
        self.loss = nn.MSELoss()
        self.rate = rate
        self.learning_rate = learning_rate
        self.mbfx = MultiBandFX(fx, 1)
        self.mbfx_layer = MBFxLayer(self.mbfx, self.rate, param_range, fake_num_bands=1)
        self.param_range = param_range
        self.loss_weights = [1, 0]
        self.audiologs = audiologs
        self.out_of_domain = out_of_domain

    def forward(self, x, *args, **kwargs) -> Any:
        out = self.norm(x)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.activation(out)
        return out

    def training_step(self, batch, batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        if self.out_of_domain:
            raise NotImplementedError
        else:
            _, _, feat, label = batch
            pred = self.forward(feat)
            loss = self.loss(pred, label)
            self.logger.experiment.add_scalar("Param_loss/Train", loss, global_step=self.global_step)
            scalars = {}
            for (i, val) in enumerate(torch.mean(torch.abs(pred - label), 0)):
                scalars[f'{i}'] = val
            self.logger.experiment.add_scalars("Param_distance/Train", scalars, global_step=self.global_step)
            return loss

    def validation_step(self, batch, batch_idx, dataloader_idx, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        # TODO: Copy from AutoFXRESNET
        # TODO: Add second dataloader for testing in and out-of-domain while training
        if dataloader_idx == 1:
            pass
        else:
            clean, processed, feat, label = batch
            clean = clean.to("cpu")
            batch_size = processed.shape[0]
            pred = self.forward(feat)
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
            if self.loss_weights[1] != 0:
                pass
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
            total_loss = loss
            self.logger.experiment.add_scalar("Total_loss/test", total_loss, global_step=self.global_step)
            return total_loss

    def on_validation_end(self) -> None:
        clean, processed, feat, label = next(iter(self.trainer.val_dataloaders[0]))
        pred = self.forward(feat.to(self.device))
        pred = pred.to("cpu")
        rec = torch.zeros(clean.shape[0], clean.shape[-1], device=self.device)
        for (i, snd) in enumerate(clean):
            # self.mbfx_layer.params = nn.Parameter(pred[i])
            rec[i] = self.mbfx_layer.forward(snd, pred[i])
        for l in range(self.audiologs):
            self.logger.experiment.add_audio(f"Audio/{l}/Original", processed[l] / torch.max(torch.abs(processed[l])),
                                             sample_rate=self.rate, global_step=self.global_step)
            self.logger.experiment.add_audio(f"Audio/{l}/Matched", rec[l] / torch.max(torch.abs(rec[l])),
                                             sample_rate=self.rate, global_step=self.global_step)
            self.logger.experiment.add_text(f"Audio/{l}/Predicted_params", str(pred[l]), global_step=self.global_step)
            self.logger.experiment.add_text(f"Audio/{l}/Original_params", str(label[l]), global_step=self.global_step)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
