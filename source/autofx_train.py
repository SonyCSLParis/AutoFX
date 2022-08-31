import pathlib

from carbontracker.tracker import CarbonTracker
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
from source.models.autofx_nogru import AutoFX
from source.models.autofx_resnet import AutoFX as AFXResNet
from source.data.mbfx_dataset import MBFXDataset
from source.data.idmt_dataset import IDMTDataset
import pedalboard as pdb

DATASET_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_dry_22050")
PROCESSED_PATH = pathlib.Path("/home/alexandre/dataset/mbfx_chorus_guitar_mono_lite")
# PROCESSED_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_modulation_22050")
NUM_EPOCHS = 80

logger = TensorBoardLogger("/home/alexandre/logs", name="resnet_chorus")

# tracker = None
dataset = MBFXDataset(PROCESSED_PATH / 'params.csv', DATASET_PATH, PROCESSED_PATH, rate=22050)
# dataset = IDMTDataset(PROCESSED_PATH / "fx2clean.csv", DATASET_PATH, PROCESSED_PATH)
train, test = random_split(dataset, [28000, 4720 * 2])
# train, test = random_split(dataset, [2808, 936])
# train, test = random_split(dataset, [5000, 2488])

cnn = AFXResNet(fx=[pdb.Chorus], num_bands=1, total_num_bands=1, learning_rate=0.001,
                param_range=[(0.1, 10), (0, 1), (0, 20), (0, 1), (0, 1)],
                tracker=False, out_of_domain=False, audiologs=6,
                conv_k=[5, 5, 5, 5, 5], conv_ch=[64, 64, 64, 64, 64], mel_spectro=True,
                conv_stride=[2, 2, 2, 2, 2], loss_stamps=[40, 160])

checkpoint_callback = ModelCheckpoint(every_n_epochs=5, save_top_k=-1)
trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=NUM_EPOCHS, auto_select_gpus=True,
                     callbacks=[checkpoint_callback], log_every_n_steps=100, detect_anomaly=True)
trainer.fit(cnn, DataLoader(train, batch_size=32, num_workers=6, shuffle=True),
            DataLoader(test, batch_size=32, num_workers=6, shuffle=True),
            ckpt_path="/home/alexandre/logs/resnet_chorus/version_50/checkpoints/epoch=39-step=35000.ckpt")
# trainer = pl.Trainer()
# trainer.fit(cnn, DataLoader(train, batch_size=64, num_workers=4, shuffle=True),
#            DataLoader(test, batch_size=64, num_workers=4, shuffle=True),
#            ckpt_path="/home/alexandre/logs/Full_train/version_6/checkpoints/epoch=49-step=21900.ckpt")
