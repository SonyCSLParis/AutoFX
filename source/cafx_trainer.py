import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI
from source.models.CAFx import CAFx
from source.data.datamodules import FeaturesDataModule

cli = LightningCLI(CAFx, FeaturesDataModule)



