import pathlib

from light_network import LightNetwork
from datamodules import FeaturesDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
import pedalboard as pdb

CLEAN_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_dry_22050_cut")
PROCESSED_PATH = pathlib.Path("/home/alexandre/dataset/modulation_guitar_mono_cut")
OUT_OF_DOMAIN_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_modulation_22050_cut")
NUM_EPOCHS = 500
PARAM_RANGE = [(0.1, 10), (0, 1), (0, 20), (0, 1), (0, 1)]

logger = TensorBoardLogger("/home/alexandre/logs", name="lite_modulation")

datamodule = FeaturesDataModule(CLEAN_PATH, PROCESSED_PATH, OUT_OF_DOMAIN_PATH)

lite_network = LightNetwork(28, 128, 5, [pdb.Chorus], PARAM_RANGE)
checkpoint_callback = ModelCheckpoint(every_n_epochs=5, save_top_k=-1)

trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=NUM_EPOCHS, auto_select_gpus=True,
                     callbacks=[checkpoint_callback], log_every_n_steps=100, detect_anomaly=True)
trainer.fit(lite_network, datamodule=datamodule)