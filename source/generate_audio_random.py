import os

import auraloss
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
CHECKPOINT = "/home/alexandre/logs/dmd_12aoutFeat/lightning_logs/version_1/checkpoints/epoch=19-step=27920.ckpt"
# CHECKPOINT = "/home/alexandre/logs/resnet_chorus/version_50/checkpoints/epoch=39-step=35000.ckpt"
CLEAN_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_dry_22050_cut")
PROCESSED_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_modulation_delay_distortion_22050_cut")
IN_DOMAIN_PATH = pathlib.Path("/home/alexandre/dataset/modulation_delay_distortion_guitar_mono_cut")

SAVE_PATH = pathlib.Path("/home/alexandre/Music/perceptual_exp/param-feat_no_finetuning")
if not SAVE_PATH.exists():
    os.mkdir(SAVE_PATH)

mrstft_fft = [64, 128, 256, 512, 1024, 2048]
mrstft_hop = [16, 32, 64, 128, 256, 512]

mrstft = auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=mrstft_fft,
                                               hop_sizes=mrstft_hop,
                                               win_lengths=mrstft_fft,
                                               w_phs=1,
                                               w_lin_mag=0,
                                               w_sc=0)
spectral_loss = mrstft
temporal_loss = torch.nn.MSELoss()

OUT_OF_DOMAIN = True

model = AutoFX.load_from_checkpoint(CHECKPOINT, num_bands=1, cond_feat=48,
                                    param_range_delay=PARAM_RANGE_DELAY,
                                    param_range_modulation=PARAM_RANGE_MODULATION,
                                    param_range_disto=PARAM_RANGE_DISTORTION, aggregated_classes=True,
                                    disable_feat=False)

datamodule = FeaturesDataModule(CLEAN_PATH, IN_DOMAIN_PATH, PROCESSED_PATH, seed=25,
                                conditioning=True, classes2keep=[0, 1, 2], out_of_domain=OUT_OF_DOMAIN,
                                out_scaler_mean=model.scaler.mean.detach().cpu().clone(),
                                out_scaler_std=model.scaler.std.detach().cpu().clone(),
                                in_scaler_mean=model.scaler.mean.detach().cpu().clone(),
                                in_scaler_std=model.scaler.std.detach().cpu().clone(),
                                batch_size=128, return_file_name=True, aggregated_classes=True,
                                out_csv_name="data_full.csv", csv_name="data_simple2.csv",
                                fx_feat=True, clf_feat=False
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
    clean, processed, feat, conditioning, fx_class, filename = batch
    pred = model.forward(processed, feat, conditioning)
    # print(torch.square(pred - label))
    cnt_disto, cnt_modulation, cnt_delay = 0, 0, 0
    i = 0
    while cnt_disto != 10 or cnt_delay != 10 or cnt_modulation != 10:
        print(i, cnt_modulation, cnt_delay, cnt_disto)
        if filename[i].split('-')[2][1:3] in ["31", "32", "33", "35"] and cnt_modulation != 10:
            rec = model.board_layers[0].forward(clean[i], pred[i, :5])
            rec = rec / torch.max(torch.abs(rec))
            print('max processed', torch.max(torch.abs(processed[i])))
            print('max rec', torch.max(torch.abs(rec)))
            print("Spectral loss: ", spectral_loss(processed[i], rec))
            print("temporal loss: ", temporal_loss(processed[i], rec))
            cnt_modulation += 1
            sf.write(SAVE_PATH / f"modulation_ref_{cnt_modulation}.wav", processed[i].T, 22050)
            sf.write(SAVE_PATH / f"modulation_rec_{cnt_modulation}.wav", rec.T, 22050)
        elif filename[i].split('-')[2][1:3] in ['21', '22'] and cnt_delay != 10:
            rec = model.board_layers[1].forward(clean[i], pred[i, 5:8])
            rec = rec / torch.max(torch.abs(rec))
            print('max processed', torch.max(torch.abs(processed[i])))
            print('max rec', torch.max(torch.abs(rec)))
            print("Spectral loss: ", spectral_loss(processed[i], rec))
            print("temporal loss: ", temporal_loss(processed[i], rec))
            cnt_delay += 1
            sf.write(SAVE_PATH / f"delay_ref_{cnt_delay}.wav", processed[i].T, 22050)
            sf.write(SAVE_PATH / f"delay_rec_{cnt_delay}.wav", rec.T, 22050)
        elif filename[i].split('-')[2][0] == '4' and cnt_disto != 10:
            rec = model.board_layers[2].forward(clean[i], pred[i, 8:])
            rec = rec / torch.max(torch.abs(rec))
            print('max processed', torch.max(torch.abs(processed[i])))
            print('max rec', torch.max(torch.abs(rec)))
            print("Spectral loss: ", spectral_loss(processed[i], rec))
            print("temporal loss: ", temporal_loss(processed[i], rec))
            cnt_disto += 1
            sf.write(SAVE_PATH / f"disto_ref_{cnt_disto}.wav", processed[i].T, 22050)
            sf.write(SAVE_PATH / f"disto_rec_{cnt_disto}.wav", rec.T, 22050)
        i += 1
