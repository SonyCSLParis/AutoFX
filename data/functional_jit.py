"""
Functionals to be used for simpler representation of time changing-features.
"""
import librosa.feature
import numpy as np
import torch
from numpy.typing import ArrayLike
from source.config import DATA_DICT
from scipy.stats import skew, kurtosis
import source.util as util


def f_max(arr, dim: int = -1):
    return torch.max(arr, dim=dim)[0]


def f_min(arr, dim: int = -1):
    return torch.min(arr, dim=dim)[0]


def f_avg(arr, dim: int = -1):
    return torch.mean(arr, dim=dim)


def f_std(arr, dim: int = -1):
    dev = torch.std(arr, dim=dim)
    # if torch.isnan(dev).any():
    #    raise ValueError
    return dev


def f_skew(arr, dim: int = -1):
    mean = torch.mean(arr, dim=dim, keepdim=True)
    diffs = arr - mean
    var = torch.mean(torch.pow(diffs, 2), dim=dim, keepdim=True)
    zscores = diffs / torch.sqrt(var)
    skews = torch.mean(torch.pow(zscores, 3), dim=dim)
    return skews


def f_kurt(arr, dim: int = -1):
    mean = torch.mean(arr, dim=dim, keepdim=True)
    diffs = arr - mean
    var = torch.mean(torch.pow(diffs, 2), dim=dim, keepdim=True)
    zscores = diffs / torch.sqrt(var)
    kurts = torch.mean(torch.pow(zscores, 4), dim=dim)
    return kurts


def linear_regression(feat):
    lin_coeff, lin_residual, _ = np.polyfit(np.arange(len(feat)), feat, 1, full=True)
    return lin_coeff, lin_residual


def quad_reg(feat):
    quad_coeff, quad_residual, _ = np.polyfit(np.arange(len(feat)), feat, 2, full=True)
    return quad_coeff, quad_residual


def fft_max_batch(feat, num_max: int = 1, zero_half_width: int = None, beta: int = 1000):
    dc_feat = feat - torch.mean(feat, dim=-1, keepdim=True)
    window = torch.hann_window(dc_feat.shape[-1], device=feat.device)
    windows = torch.vstack([window] * feat.shape[0])
    dc_feat_w = dc_feat * windows
    rfft = torch.fft.rfft(dc_feat_w, 1024)
    rfft = torch.abs(rfft) * 4 / 1024
    rfft[:, :16] = torch.zeros_like(rfft[:, :16])
    rfft_max, _ = torch.max(rfft, dim=-1)
    rfft_max = rfft_max[:, None]
    # rfft_norm = rfft / rfft_max
    rfft_max_bin = util.approx_argmax(rfft, beta=beta)
    if num_max > 1:
        cnt = 1
        while cnt < num_max:
            tmp1 = rfft_max_bin[:, cnt - 1] - zero_half_width
            low_bound = torch.nn.functional.relu(tmp1)
            tmp2 = rfft_max_bin[:, cnt - 1] + zero_half_width + 1
            high_bound = -torch.nn.functional.relu(-tmp2 + 513) + 513
            # zero mask batch: https://stackoverflow.com/questions/57548180/filling-torch-tensor-with-zeros-after-certain-index
            mask = torch.zeros(rfft.shape[0], rfft.shape[1] + 1, device=feat.device)
            mask[(torch.arange(rfft.shape[0]), low_bound.long())] = 1
            mask[(torch.arange(rfft.shape[0]), high_bound.long())] = -1
            mask = mask.cumsum(dim=1)[:, :-1]
            rfft = rfft * (1. - mask)
            max_val, _ = torch.max(rfft, dim=-1)
            max_bin = util.approx_argmax(rfft + 1, beta=beta)
            rfft_max = torch.cat((rfft_max, max_val[:, None]), dim=-1)
            rfft_max_bin = torch.cat((rfft_max_bin, max_bin), dim=-1)
            cnt += 1
    return rfft_max, rfft_max_bin


def fft_max(feat, num_max: int = 1, zero_half_width: int = None):
    """
    https://github.com/henrikjuergens/guitar-fx-extraction/blob/master/featextr.py
    :param feat: Features to analyse
    :param num_max: Number of fft max to retrieve;
    :param zero_half_width: Half number of bins that should be zeroed before retrieving
    a new maximum. 2*zero_half_width + 1 bins are zeroed, centered on the previous max.
    :return: Lists (possibly of len==1) of fft max and argmax.
    """
    dc_feat = feat - np.mean(feat)
    dc_feat_w = dc_feat * np.hanning(len(dc_feat))
    rfft = np.fft.rfft(dc_feat_w, 1024)
    rfft = np.abs(rfft) * 4 / 1024
    rfft[0, :16] = np.zeros(16)  # TODO: Find why?
    rfft_max = np.max(rfft)
    rfft_max_bin = np.argmax(rfft)
    rfft_max = [rfft_max]
    rfft_max_bin = [rfft_max_bin]
    if num_max > 1:
        cnt = 1
        zeros = np.zeros(2 * zero_half_width + 1)
        while cnt < num_max:
            low_bound = max(0, rfft_max_bin[cnt - 1] - zero_half_width)
            high_bound = min(513, rfft_max_bin[cnt - 1] + zero_half_width + 1)
            rfft[0, low_bound:high_bound] = zeros[:high_bound - low_bound]
            rfft_max.append(np.max(rfft))
            rfft_max_bin.append(np.argmax(rfft))
            cnt += 1
    return rfft_max, rfft_max_bin


def estim_derivative(feat, dim: int = -1):
    prepend = torch.zeros_like(feat)
    prepend = torch.sum(prepend, dim=dim, keepdim=True)
    return torch.diff(feat, dim=dim, prepend=prepend)


def feat_vector(feat: dict, pitch: float) -> dict:
    """
    Returns a dict of all the functionals listed in config.DATA_DICT from feat, a dictionary representing the
    spectral features attribute of a SoundSample object.
    :param feat: Dictionary of the features;
    :param pitch: Pitch in Hertz of the note. Necessary for normalization of some features.
    """
    out = DATA_DICT
    cent = feat['centroid']
    out['cent_avg'] = f_avg(cent)
    out['cent_std'] = f_std(cent)
    out['cent_skw'] = f_skew(cent)
    out['cent_krt'] = f_kurt(cent)
    out['cent_min'] = f_min(cent)
    out['cent_max'] = f_max(cent)
    spread = feat['spread']
    out['spread_avg'] = f_avg(spread)
    out['spread_std'] = f_std(spread)
    out['spread_skw'] = f_skew(spread)
    out['spread_krt'] = f_kurt(spread)
    out['spread_min'] = f_min(spread)
    out['spread_max'] = f_max(spread)
    skew = feat['skewness']
    out['skew_avg'] = f_avg(skew)
    out['skew_std'] = f_std(skew)
    out['skew_skw'] = f_skew(skew)
    out['skew_krt'] = f_kurt(skew)
    out['skew_min'] = f_min(skew)
    out['skew_max'] = f_max(skew)
    kurt = feat['kurtosis']
    out['kurt_avg'] = f_avg(kurt)
    out['kurt_std'] = f_std(kurt)
    out['kurt_skw'] = f_skew(kurt)
    out['kurt_krt'] = f_kurt(kurt)
    out['kurt_min'] = f_min(kurt)
    out['kurt_max'] = f_max(kurt)
    # pitch normalized
    n_cent = cent / pitch
    out['n_cent_avg'] = f_avg(n_cent)
    out['n_cent_std'] = f_std(n_cent)
    out['n_cent_skw'] = f_skew(n_cent)
    out['n_cent_krt'] = f_kurt(n_cent)
    out['n_cent_min'] = f_min(n_cent)
    out['n_cent_max'] = f_max(n_cent)
    n_spread = spread / pitch
    out['n_spread_avg'] = f_avg(n_spread)
    out['n_spread_std'] = f_std(n_spread)
    out['n_spread_skw'] = f_skew(n_spread)
    out['n_spread_krt'] = f_kurt(n_spread)
    out['n_spread_min'] = f_min(n_spread)
    out['n_spread_max'] = f_max(n_spread)
    n_skew = skew / pitch
    out['n_skew_avg'] = f_avg(n_skew)
    out['n_skew_std'] = f_std(n_skew)
    out['n_skew_skw'] = f_skew(n_skew)
    out['n_skew_krt'] = f_kurt(n_skew)
    out['n_skew_min'] = f_min(n_skew)
    out['n_skew_max'] = f_max(n_skew)
    n_kurt = kurt / pitch
    out['n_kurt_avg'] = f_avg(n_kurt)
    out['n_kurt_std'] = f_std(n_kurt)
    out['n_kurt_skw'] = f_skew(n_kurt)
    out['n_kurt_krt'] = f_kurt(n_kurt)
    out['n_kurt_min'] = f_min(n_kurt)
    out['n_kurt_max'] = f_max(n_kurt)
    flux = feat['flux']
    out['flux_avg'] = f_avg(flux)
    out['flux_std'] = f_std(flux)
    out['flux_skw'] = f_skew(flux)[0]
    out['flux_krt'] = f_kurt(flux)[0]
    out['flux_min'] = f_min(flux)
    out['flux_max'] = f_max(flux)
    rolloff = feat['rolloff']
    out['rolloff_avg'] = f_avg(rolloff)
    out['rolloff_std'] = f_std(rolloff)
    out['rolloff_skw'] = f_skew(rolloff)[0]
    out['rolloff_krt'] = f_kurt(rolloff)[0]
    out['rolloff_min'] = f_min(rolloff)
    out['rolloff_max'] = f_max(rolloff)
    slope = feat['slope']
    out['slope_avg'] = f_avg(slope)
    out['slope_std'] = f_std(slope)
    out['slope_skw'] = f_skew(slope)[0]
    out['slope_krt'] = f_kurt(slope)[0]
    out['slope_min'] = f_min(slope)
    out['slope_max'] = f_max(slope)
    flat = feat['flatness']
    out['flat_avg'] = f_avg(flat)
    out['flat_std'] = f_std(flat)
    out['flat_skw'] = f_skew(flat)[0]
    out['flat_krt'] = f_kurt(flat)[0]
    out['flat_min'] = f_min(flat)
    out['flat_max'] = f_max(flat)
    # hi-passed aka delta features
    cent_hp = util.hi_pass(cent)
    out['cent_hp_avg'] = f_avg(cent_hp)
    out['cent_hp_std'] = f_std(cent_hp)
    out['cent_hp_skw'] = f_skew(cent_hp)
    out['cent_hp_krt'] = f_kurt(cent_hp)
    out['cent_hp_min'] = f_min(cent_hp)
    out['cent_hp_max'] = f_max(cent_hp)
    spread_hp = util.hi_pass(spread)
    out['spread_hp_avg'] = f_avg(spread_hp)
    out['spread_hp_std'] = f_std(spread_hp)
    out['spread_hp_skw'] = f_skew(spread_hp)
    out['spread_hp_krt'] = f_kurt(spread_hp)
    out['spread_hp_min'] = f_min(spread_hp)
    out['spread_hp_max'] = f_max(spread_hp)
    skew_hp = util.hi_pass(skew)
    out['skew_hp_avg'] = f_avg(skew_hp)
    out['skew_hp_std'] = f_std(skew_hp)
    out['skew_hp_skw'] = f_skew(skew_hp)
    out['skew_hp_krt'] = f_kurt(skew_hp)
    out['skew_hp_min'] = f_min(skew_hp)
    out['skew_hp_max'] = f_max(skew_hp)
    kurt_hp = util.hi_pass(kurt)
    out['kurt_hp_avg'] = f_avg(kurt_hp)
    out['kurt_hp_std'] = f_std(kurt_hp)
    out['kurt_hp_skw'] = f_skew(kurt_hp)
    out['kurt_hp_krt'] = f_kurt(kurt_hp)
    out['kurt_hp_min'] = f_min(kurt_hp)
    out['kurt_hp_max'] = f_max(kurt_hp)
    n_cent_hp = util.hi_pass(n_cent)
    out['n_cent_hp_avg'] = f_avg(n_cent_hp)
    out['n_cent_hp_std'] = f_std(n_cent_hp)
    out['n_cent_hp_skw'] = f_skew(n_cent_hp)
    out['n_cent_hp_krt'] = f_kurt(n_cent_hp)
    out['n_cent_hp_min'] = f_min(n_cent_hp)
    out['n_cent_hp_max'] = f_max(n_cent_hp)
    n_spread_hp = util.hi_pass(n_spread)
    out['n_spread_hp_avg'] = f_avg(n_spread_hp)
    out['n_spread_hp_std'] = f_std(n_spread_hp)
    out['n_spread_hp_skw'] = f_skew(n_spread_hp)
    out['n_spread_hp_krt'] = f_kurt(n_spread_hp)
    out['n_spread_hp_min'] = f_min(n_spread_hp)
    out['n_spread_hp_max'] = f_max(n_spread_hp)
    n_skew_hp = util.hi_pass(n_skew)
    out['n_skew_hp_avg'] = f_avg(n_skew_hp)
    out['n_skew_hp_std'] = f_std(n_skew_hp)
    out['n_skew_hp_skw'] = f_skew(n_skew_hp)
    out['n_skew_hp_krt'] = f_kurt(n_skew_hp)
    out['n_skew_hp_min'] = f_min(n_skew_hp)
    out['n_skew_hp_max'] = f_max(n_skew_hp)
    n_kurt_hp = util.hi_pass(n_kurt)
    out['n_kurt_hp_avg'] = f_avg(n_kurt_hp)
    out['n_kurt_hp_std'] = f_std(n_kurt_hp)
    out['n_kurt_hp_skw'] = f_skew(n_kurt_hp)
    out['n_kurt_hp_krt'] = f_kurt(n_kurt_hp)
    out['n_kurt_hp_min'] = f_min(n_kurt_hp)
    out['n_kurt_hp_max'] = f_max(n_kurt_hp)
    flux_hp = util.hi_pass(flux)
    out['flux_hp_avg'] = f_avg(flux_hp)
    out['flux_hp_std'] = f_std(flux_hp)
    out['flux_hp_skw'] = f_skew(flux_hp)[0]
    out['flux_hp_krt'] = f_kurt(flux_hp)[0]
    out['flux_hp_min'] = f_min(flux_hp)
    out['flux_hp_max'] = f_max(flux_hp)
    rolloff_hp = util.hi_pass(rolloff)
    out['rolloff_hp_avg'] = f_avg(rolloff_hp)
    out['rolloff_hp_std'] = f_std(rolloff_hp)
    out['rolloff_hp_skw'] = f_skew(rolloff_hp)[0]
    out['rolloff_hp_krt'] = f_kurt(rolloff_hp)[0]
    out['rolloff_hp_min'] = f_min(rolloff_hp)
    out['rolloff_hp_max'] = f_max(rolloff_hp)
    slope_hp = util.hi_pass(slope)
    out['slope_hp_avg'] = f_avg(slope_hp)
    out['slope_hp_std'] = f_std(slope_hp)
    out['slope_hp_skw'] = f_skew(slope_hp)[0]
    out['slope_hp_krt'] = f_kurt(slope_hp)[0]
    out['slope_hp_min'] = f_min(slope_hp)
    out['slope_hp_max'] = f_max(slope_hp)
    flat_hp = util.hi_pass(flat)
    out['flat_hp_avg'] = f_avg(flat_hp)
    out['flat_hp_std'] = f_std(flat_hp)
    out['flat_hp_skw'] = f_skew(flat_hp)[0]
    out['flat_hp_krt'] = f_kurt(flat_hp)[0]
    out['flat_hp_min'] = f_min(flat_hp)
    out['flat_hp_max'] = f_max(flat_hp)
    return out
