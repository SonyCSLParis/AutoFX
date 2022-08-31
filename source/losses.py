"""
Implementation based on Martinez Ramirez et al., 2021
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import auraloss
import torchaudio.functional


def compute_time_delay(original, pred, max_shift: int = 100):
    original_fft = torch.fft.fft(original)
    batch_size = original.shape[0]
    pred_fft = torch.fft.fft(pred)
    xcorr_fft = torch.multiply(torch.conj(original_fft), pred_fft)
    xcorr = torch.fft.ifft(xcorr_fft)
    time_shift = torch.argmax(torch.abs(xcorr), dim=-1)
    for i in range(batch_size):
        if time_shift[i] > max_shift:
            time_shift[i] = 0
    return time_shift


def time_align_signals(original, pred):
    time_shift = compute_time_delay(original, pred)
    batch_size = original.shape[0]
    original_shifted = torch.zeros_like(original)
    pred_shifted = torch.zeros_like(pred)
    for i in range(batch_size):
        original_shifted[i, :-time_shift[i]] = original[i, :-time_shift[i]]
        pred_shifted[i, time_shift[i]:] = pred[i, time_shift[i]:]
    return original_shifted, pred_shifted


def custom_time_loss(output, target, aligned: bool = False):
    if not aligned:
        target_aligned, output_aligned = time_align_signals(target, output)
    else:
        target_aligned = target
        output_aligned = output
    loss_plus = F.l1_loss(target_aligned, output_aligned)
    loss_minus = F.l1_loss(target_aligned, -1 * output_aligned)
    return torch.min(loss_minus, loss_plus)


def custom_spectral_loss(output, target, fft_size: int = 1024):
    target_aligned, output_aligned = time_align_signals(target, output)
    target_fft = torch.fft.fft(target_aligned, fft_size=fft_size, dim=-1)
    output_fft = torch.fft.fft(output_aligned, fft_size=fft_size, dim=-1)
    target_mag = torch.abs(target_fft)
    output_mag = torch.abs(output_fft)
    loss = F.mse_loss(target_mag, output_mag)
    log_loss = F.mse_loss(torch.log(target_mag), torch.log(output_mag))
    return loss + log_loss


def custom_loss(output, target):
    time_loss = custom_time_loss(output, target)
    spectral_loss = custom_spectral_loss(output, target)
    return 10 * time_loss + spectral_loss


def pitch_loss(output, target, rate, power: int = 1):
    pitch_output = torchaudio.functional.detect_pitch_frequency(output, rate)
    pitch_target = torchaudio.functional.detect_pitch_frequency(target, rate)
    return torch.pow(torch.abs(pitch_target - pitch_output), power)


class PitchLoss(nn.Module):
    def __init__(self, rate, power: int = 1):
        super().__init__()
        self.power = power
        self.rate = rate

    def forward(self, output, target):
        return torch.mean(pitch_loss(output, target, self.rate, self.power))
