"""
Convolutional Neural Network classifier on spectrograms for comparison.
"""

import os
import pathlib
from typing import Any, Optional

from carbontracker.tracker import CarbonTracker

import numpy as np
import torch
import torchmetrics
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
import sys

import util

sys.path.append('..')
from spectrogram_dataset import SpectroDataset

DATASET_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_one_folder")


class CnnClassifier(pl.LightningModule):
    def __init__(self, tracker: CarbonTracker=None):
        super().__init__()
        self.model = nn.Sequential(nn.Conv2d(1, 6, 5),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2),
                                   nn.Conv2d(6, 16, 5),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2),
                                   nn.Flatten(1),
                                   nn.Linear(21056, 120),
                                   nn.ReLU(),
                                   nn.Linear(120, 84),
                                   nn.ReLU(),
                                   nn.Linear(84, 11))
        self.loss = nn.CrossEntropyLoss()
        self.tracker = tracker

    def forward(self, x, *args, **kwargs) -> Any:
        return self.model(x)

    def training_step(self, batch, batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        data, label = batch
        pred = self.model(data)
        loss = self.loss(pred, label)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        data, label = batch
        pred = self.model(data)
        self.log("Accuracy", torchmetrics.functional.accuracy(pred, label))
        self.log("Precision", torchmetrics.functional.precision(pred, label))
        self.log("Recall", torchmetrics.functional.recall(pred, label))

    def on_train_start(self) -> None:
        dataiter = iter(self.trainer.val_dataloaders[0])
        spectro, labels = dataiter.next()
        self.logger.experiment.add_images("Input spectrograms", spectro)
        self.logger.experiment.add_text('Classes:', str(labels))

    def on_train_epoch_start(self) -> None:
        if self.tracker is not None:
            self.tracker.epoch_start()

    def on_train_epoch_end(self) -> None:
        if self.tracker is not None:
            self.tracker.epoch_end()
        dataiter = iter(self.trainer.val_dataloaders[0])
        spectro, labels = dataiter.next()
        pred = self.model(spectro.cuda())  # TODO: Send to correct device in a cleaner way
        cm = torchmetrics.functional.confusion_matrix(pred, labels.cuda(), len(util.CLASSES))  # TODO: idem
        self.logger.experiment.add_figure("Confusion Matrix",
                                          util.make_confusion_matrix(cm.cpu().detach().numpy(),
                                                                     categories=util.CLASSES, percent=False),
                                          global_step=self.global_step)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return optimizer


logger = TensorBoardLogger("lightning_logs")
tracker = CarbonTracker(epochs=1000, epochs_before_pred=10, monitor_epochs=10, log_dir=logger.log_dir, verbose=2)
dataset = SpectroDataset(DATASET_PATH / 'labels.csv', DATASET_PATH, idmt=True, rate=44100)
train, test = random_split(dataset, [15000, 5592])

cnn = CnnClassifier(tracker=tracker)
trainer = pl.Trainer(gpus=1, logger=logger)
trainer.fit(cnn, DataLoader(train, batch_size=32, num_workers=4), DataLoader(test, batch_size=256, num_workers=8))
