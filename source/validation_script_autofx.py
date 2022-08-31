import tqdm

from source.models.AutoFX import AutoFX
import pedalboard as pdb
import pathlib
from source.data.idmt_dataset import IDMTDataset
from source.data.mbfx_dataset import MBFXDataset
import sounddevice as sd
from source.data.datasets import FeatureInDomainDataset, FeatureOutDomainDataset
from source.data.datamodules import FeaturesDataModule
import torch
import soundfile as sf

WRITE_AUDIO = True
PARAM_RANGE_DISTORTION = [(0, 60),
                          (50, 500), (-10, 10), (0.5, 2),
                          (500, 2000), (-10, 10), (0.5, 2)]
PARAM_RANGE_DELAY = [(0, 1), (0, 1), (0, 1)]
PARAM_RANGE_MODULATION = [(0.1, 10), (0, 1), (0, 20), (0, 1), (0, 1)]
CHECKPOINT = "/home/alexandre/logs/dmd_weekend/lightning_logs/version_0/checkpoints/epoch=24-step=32700.ckpt"
# CHECKPOINT = "/home/alexandre/logs/resnet_chorus/version_50/checkpoints/epoch=39-step=35000.ckpt"
CLEAN_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_dry_22050_cut")
PROCESSED_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_modulation_delay_distortion_22050_cut")
IN_DOMAIN_PATH = pathlib.Path("/home/alexandre/dataset/modulation_delay_distortion_guitar_mono_cut")

OUT_OF_DOMAIN = True

model = AutoFX.load_from_checkpoint(CHECKPOINT, num_bands=1, cond_feat=48,
                                    param_range_delay=PARAM_RANGE_DELAY,
                                    param_range_modulation=PARAM_RANGE_MODULATION,
                                    param_range_disto=PARAM_RANGE_DISTORTION)

datamodule = FeaturesDataModule(CLEAN_PATH, IN_DOMAIN_PATH, PROCESSED_PATH, seed=2,
                                conditioning=True, classes2keep=[0, 1, 2], out_of_domain=OUT_OF_DOMAIN,
                                out_scaler_mean=model.scaler.mean.detach().cpu().clone(),
                                out_scaler_std=model.scaler.std.detach().cpu().clone(),
                                in_scaler_mean=model.scaler.mean.detach().cpu().clone(),
                                in_scaler_std=model.scaler.std.detach().cpu().clone()
                                )
datamodule.setup()


# model = AutoFX.load_from_checkpoint(CHECKPOINT,
#                                    fx=[pdb.Chorus], num_bands=1,
#                                    param_range=PARAM_RANGE)

model.out_of_domain = OUT_OF_DOMAIN
model.freeze()

if WRITE_AUDIO:
    datamodule.out_of_domain = OUT_OF_DOMAIN
    batch = next(iter(datamodule.val_dataloader()))
    clean, processed, feat, conditioning, fx_class = batch
    pred = model.forward(processed, feat, conditioning)
    # print(torch.square(pred - label))
    toSave = torch.zeros((1, 64 * 35000))
    for i in tqdm.tqdm(range(32)):
        if fx_class[i] == 0:
            rec = model.board_layers[0].forward(clean[i], pred[i, :5])
        elif fx_class[i] == 1:
            rec = model.board_layers[1].forward(clean[i], pred[i, 5:8])
        elif fx_class[i] == 2:
            rec = model.board_layers[2].forward(clean[i], pred[i, 8:])
        toSave[0, (2 * i) * 35000:(2 * i + 1) * 35000] = processed[i]
        toSave[0, (2 * i + 1) * 35000:2 * (i + 1) * 35000] = rec / (torch.max(torch.abs(rec)))
    sf.write("/home/alexandre/Music/dmd_feat_ood_fix.wav", toSave.T, 22050)
