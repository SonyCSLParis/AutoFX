import pathlib

from carbontracker.tracker import CarbonTracker
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
from autofx_nogru import AutoFX
from autofx_resnet import AutoFX as AFXResNet
from mbfx_dataset import MBFXDataset
from idmt_dataset import IDMTDataset
import pedalboard as pdb

out_of_domain = True

NUM_EPOCHS = 50

DATASET_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_dry_22050")
if out_of_domain:
    PROCESSED_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_modulation_22050")
    dataset = IDMTDataset(PROCESSED_PATH / "fx2clean.csv", DATASET_PATH, PROCESSED_PATH)
    train, test = random_split(dataset, [5000, 2488])
logger = TensorBoardLogger("/home/alexandre/logs", name="resnet_chorus")

cnn = AFXResNet(fx=[pdb.Chorus], num_bands=1, total_num_bands=1, learning_rate=0.01,
                param_range=[(0.1, 10), (0, 1), (0, 20), (0, 1), (0, 1)],
                tracker=False, out_of_domain=True, audiologs=6,
                conv_k=[5, 5, 5, 5, 5], conv_ch=[64, 64, 64, 64, 64], mel_spectro=True,
                conv_stride=[2, 2, 2, 2, 2], loss_stamps=[10, 25])


checkpoint_callback = ModelCheckpoint(every_n_epochs=1, save_top_k=-1)
trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=NUM_EPOCHS, auto_select_gpus=True,
                     callbacks=[checkpoint_callback], log_every_n_steps=100, detect_anomaly=True)


trainer.validate(cnn, DataLoader(train, batch_size=32, num_workers=6),
                 ckpt_path="/home/alexandre/logs/resnet_chorus/version_34/checkpoints/epoch=9-step=17500.ckpt")
