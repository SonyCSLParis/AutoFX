import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI
from source.models.AutoFX import AutoFX
from source.data.datamodules import FeaturesDataModule

cli = LightningCLI(AutoFX, FeaturesDataModule)



