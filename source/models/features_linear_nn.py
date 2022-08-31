import pathlib
import pedalboard as pdb
from typing import List, Tuple, Any, Optional

import auraloss
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

from source.data.features_dataset import FeaturesDataset
from mbfx_layer import MBFxLayer
from multiband_fx import MultiBandFX


class AutoFxLite(pl.LightningModule):
    def __init__(self, fx, num_bands: int, param_range: List[Tuple], in_features: int = 144, tracker: bool = False,
                 rate: int = 22050, file_size: int = 44100, total_num_bands: int = None,
                 audiologs: int = 4, loss_weights: list[float] = [1, 1],
                 mrstft_fft: list[int] = [256, 512, 1024, 2048],
                 mrstft_hop: list[int] = [64, 128, 256, 512],
                 learning_rate: float = 0.001, out_of_domain: bool = False,
                 spectro_power: int = 2, mel_spectro: bool = True, mel_num_bands: int = 128,
                 loss_stamps: list = None, device=torch.device('cuda'),
                 *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if total_num_bands is None:
            total_num_bands = num_bands
        self.total_num_bands = total_num_bands
        self.mbfx = MultiBandFX(fx, total_num_bands, device=torch.device('cpu'))
        self.num_params = num_bands * len(self.mbfx.settings[0])
        self.in_features = in_features
        self.activation = nn.Sigmoid()
        self.learning_rate = learning_rate
        self.loss = nn.L1Loss()
        self.tracker_flag = tracker
        self.tracker = None
        self.mrstft = auraloss.freq.MultiResolutionSTFTLoss(mrstft_fft,
                                                            mrstft_hop,
                                                            mrstft_fft,
                                                            scale=None,
                                                            n_bins=64,
                                                            sample_rate=rate,
                                                            device=device)  # TODO: Manage device properly
        # self.spectral_loss = auraloss.time.ESRLoss()
        self.spectral_loss = self.mrstft
        self.num_bands = num_bands
        self.param_range = param_range
        self.rate = rate
        self.out_of_domain = out_of_domain
        self.loss_weights = loss_weights
        self.audiologs = audiologs
        self.tmp = None
        self.loss_stamps = loss_stamps
        self.mbfx_layer = MBFxLayer(self.mbfx, self.rate, self.param_range, fake_num_bands=self.num_bands)
        self.model = nn.Sequential(
            nn.LayerNorm([in_features]),
            nn.Linear(in_features, in_features, bias=True),
            nn.Dropout(),
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, in_features, bias=True),
            nn.Dropout(),
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, in_features, bias=True),
            nn.Dropout(),
            nn.BatchNorm1d(in_features),
            # nn.LeakyReLU(inplace=True),
            nn.Linear(in_features, in_features, bias=True),
            nn.Dropout(),
            nn.BatchNorm1d(in_features),
            # nn.LeakyReLU(inplace=True),
            nn.Linear(in_features, in_features, bias=True),
            nn.Dropout(),
            nn.BatchNorm1d(in_features),
            # nn.LeakyReLU(inplace=True),
            nn.Linear(in_features, in_features // 2, bias=True),
            nn.Dropout(),
            nn.BatchNorm1d(in_features//2),
            # nn.LeakyReLU(inplace=True),
            nn.Linear(in_features // 2, self.num_params, bias=True)
        )

    def forward(self, x, *args, **kwargs) -> Any:
        out1 = self.model(x)
        out2 = self.activation(out1)
        return out2

    def training_step(self, batch, batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        # TODO: Accept spectral loss
        self.tmp = False
        _, _, label, features = batch
        pred = self.forward(features)
        loss = self.loss(pred, label)
        self.logger.experiment.add_scalar("Param_loss/Train", loss, global_step=self.global_step)
        scalars = {}
        for (i, val) in enumerate(torch.mean(torch.abs(pred - label), 0)):
            scalars[f'{i}'] = val
        self.logger.experiment.add_scalars("Param_distance/Train", scalars, global_step=self.global_step)
        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        clean, processed, label, features = batch
        clean = clean.to("cpu")
        batch_size = processed.shape[0]
        pred = self.forward(features)
        loss = self.loss(pred, label)
        self.logger.experiment.add_scalar("Param_loss/test", loss, global_step=self.global_step)
        scalars = {}
        for (i, val) in enumerate(torch.mean(torch.abs(pred - label), 0)):
            scalars[f'{i}'] = val
        self.logger.experiment.add_scalars("Param_distance/test", scalars, global_step=self.global_step)
        pred = pred.to("cpu")
        rec = torch.zeros(batch_size, clean.shape[-1], device=self.device)  # TODO: fix hardcoded value
        if not self.tmp:
            self.tmp = True
            for (i, snd) in enumerate(clean):
                if i > self.audiologs:
                    break
                self.mbfx_layer.params = nn.Parameter(pred[i])
                rec[i] = self.mbfx_layer.forward(snd)
            for l in range(self.audiologs):
                self.logger.experiment.add_audio(f"Audio/{l}/Original", processed[l] / torch.max(torch.abs(processed[l])),
                                                 sample_rate=self.rate, global_step=self.global_step)
                self.logger.experiment.add_audio(f"Audio/{l}/Matched", rec[l] / torch.max(torch.abs(rec[l])),
                                                 sample_rate=self.rate, global_step=self.global_step)
                self.logger.experiment.add_text(f"Audio/{l}/Matched_params", str(pred[l]), global_step=self.global_step)
                self.logger.experiment.add_text(f"Audio/{l}/Original_params", str(label[l]),
                                                global_step=self.global_step)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)  # TODO: Remove hardcoded values
        return optimizer


if __name__ == '__main__':
    DATASET_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_dry_22050")
    PROCESSED_PATH = pathlib.Path("/home/alexandre/dataset/mbfx_disto_guitar_mono_int")
    FEATURES_PATH = pathlib.Path("/home/alexandre/dataset/mbfx_disto_guitar_mono_int_features/dataset.csv")
    # PROCESSED_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_non-linear_22050")
    NUM_EPOCHS = 100

    logger = TensorBoardLogger("/home/alexandre/logs", name="features_linear")

    # tracker = None
    dataset = FeaturesDataset(PROCESSED_PATH / 'params.csv', FEATURES_PATH, DATASET_PATH, PROCESSED_PATH, rate=22050)
    # dataset = IDMTDataset(PROCESSED_PATH / "fx2clean.csv", DATASET_PATH, PROCESSED_PATH)
    train, test = random_split(dataset, [28000, 4720 * 2])
    # train, test = random_split(dataset, [2808, 936])

    nn_linear = AutoFxLite(fx=[pdb.Distortion(0), pdb.Gain(0)], num_bands=4, total_num_bands=4,
                    param_range=[(10, 60), (-10, 10)], audiologs=6, learning_rate=0.1)

    checkpoint_callback = ModelCheckpoint(every_n_epochs=5, save_top_k=-1)
    trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=NUM_EPOCHS, auto_select_gpus=True,
                         callbacks=[checkpoint_callback], log_every_n_steps=100, track_grad_norm=2, detect_anomaly=True)
    trainer.fit(nn_linear, DataLoader(train, batch_size=256, num_workers=6, shuffle=True),
                DataLoader(test, batch_size=256, num_workers=6, shuffle=True),
                ckpt_path="/home/alexandre/logs/features_linear/version_4/checkpoints/epoch=49-step=5500.ckpt")
    # trainer = pl.Trainer()
    # trainer.fit(cnn, DataLoader(train, batch_size=64, num_workers=4, shuffle=True),
    #            DataLoader(test, batch_size=64, num_workers=4, shuffle=True),
    #            ckpt_path="/home/alexandre/logs/Full_train/version_6/checkpoints/epoch=49-step=21900.ckpt")
