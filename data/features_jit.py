"""
Copy of features with modifications to make the jitable.
Functions to extract the relevant features,
as mentioned in Stein et al. Automatic Detection of Audio effects [...] 2010.
References:
    [1]: Geoffroy Peeters, Technical report of the CUIDADO project, 2003
    [2]: Tae Hong Park, Towards automatic musical instrument timbre recognition, 2004 (PhD thesis)
"""
import librosa
import numpy as np
import torch
import torchaudio.functional
import source.util as util


def _geom_mean(arr, dim: int = -1):
    """
    Compute the geometrical mean of an array through log conversion to avoid overflow.
    """
    return torch.exp(torch.mean(torch.log(arr), dim=dim))


def spectral_centroid(mag: torch.Tensor,
                      rate: int = 1):
    """
    Spectral centroid of each frame.

    :param mag: (..., Nfft, num_frames) Magnitude spectrogram of the input signal;
    :param freq: frequency of each frequency bin, in Hertz;
    :param rate: sampling rate of the audio. Only used if freq is None. Default is 1.
    :return: (..., 1, num_frames) spectral centroid of each input frame.
    """
    batch_size, nfft, num_frames = mag.shape
    # if freq is None:
    freq = torch.linspace(0, rate / 2, nfft)
    norm_mag = mag / torch.sum(mag, dim=1, keepdim=True)
    cent = torch.sum(norm_mag * freq[None, :, None].expand(batch_size, nfft, num_frames), dim=1, keepdim=True)
    return cent


def spectral_spread(mag: torch.Tensor,
                    cent: torch.Tensor, rate: int = 1):
    """
    Spectral spread of each frame of the input signal.

    See: Geoffroy Peeters, Cuidado project Technical report, 2003.

    :param mag: (..., nfft, num_frames) Magnitude spectrogram of the input signal;
    :param stft: Complex matrix representing a Short Time Fourier Transform;
    :param cent: Array of the spectral centroid of each frame;
    :param freq: frequency of each frequency bin, in Hertz;
    :param rate: sampling rate of the audio. Only used if freq is None. Default is 1.
    :return spread: (..., 1, num_frames) spectral spread of each input frame.
    """
    batch_size, nfft, num_frames = mag.shape
    freq = torch.linspace(0, rate/2, nfft)
    norm_mag = mag / torch.sum(mag, dim=1, keepdim=True)
    cnt_freq = freq[None, :, None].expand(batch_size, -1, num_frames) - cent
    spread = torch.sum(norm_mag * torch.square(cnt_freq), dim=1, keepdim=True)
    return spread


def spectral_skewness(mag: torch.Tensor,
                      cent: torch.Tensor,
                      rate: int = 1):
    """
    Spectral skewness of each frame of the input signal.

    See: Geoffroy Peeters, Cuidado project Technical report, 2003.

    :param mag: (..., nfft, num_frames) Magnitude spectrogram of the input signal;
    :param stft: Complex matrix representing a Short Time Fourier Transform;
    :param cent: Array of the spectral centroid of each frame;
    :param freq: frequency of each frequency bin, in Hertz;
    :param rate: sampling rate of the audio. Only used if freq is None. Default is 1.
    :return skew: (..., 1, num_frames) spectral skewness of each input frame.
    """
    batch_size, nfft, num_frames = mag.shape
    freq = torch.linspace(0, rate/2, nfft)
    norm_mag = mag / torch.sum(mag, dim=1, keepdim=True)
    cnt_freq = freq[None, :, None].expand(batch_size, -1, num_frames) - cent
    skew = torch.sum(norm_mag * torch.pow(cnt_freq, 3), dim=1, keepdim=True)
    return skew


def spectral_kurtosis(mag: torch.Tensor,
                      cent: torch.Tensor, rate: int = 1):
    """
    Spectral kurtosis of each frame of the input signal.

    See: Geoffroy Peeters, Cuidado project Technical report, 2003.

    :param mag: (..., nfft, num_frames) Magnitude spectrogram of the input signal;
    :param stft: Complex matrix representing a Short Time Fourier Transform;
    :param cent: Array of the spectral centroid of each frame;
    :param freq: frequency of each frequency bin, in Hertz;
    :param rate: sampling rate of the audio. Only used if freq is None. Default is 1.
    :return kurt: (..., 1, num_frames) spectral kurtosis of each input frame.
    """
    batch_size, nfft, num_frames = mag.shape
    freq = torch.linspace(0, rate/2, nfft)
    norm_mag = mag / torch.sum(mag, dim=1, keepdim=True)
    cnt_freq = freq[None, :, None].expand(batch_size, -1, num_frames) - cent
    kurt = torch.sum(norm_mag * torch.pow(cnt_freq, 4), dim=1, keepdim=True)
    return kurt


def spectral_flux(mag, q_norm: int = 1):
    """
    Amount of frame-to-frame fluctuation in time.
    See: Tae Hong Park, Towards automatic musical instrument timbre recognition, 2004 (PhD thesis)

    :param mag: (..., Nfft, num_frames) Matrix of the frame-by-frame magnitude;
    :param q_norm: order of the q norm to use. Defaults to 1;
    :return flux: (..., 1, num_frames) matrix of the spectral flux, set to 0 for the first frame.
    """
    batch_size, nfft, num_frames = mag.shape
    diff = torch.diff(mag, n=1, dim=-1)
    flux = torch.zeros((batch_size, 1, num_frames))
    flux[:, :, 1:] = torch.pow(torch.sum(torch.pow(torch.abs(diff), q_norm),
                               dim=1, keepdim=True),
                     1 / q_norm)
    return flux


def spectral_rolloff(mag, threshold: float = 0.95,
                     rate: int = 1):
    """
    The spectral roll-off point is the frequency so that threshold% of the signal energy is contained
    below that frequency.
    See: Geoffroy Peeter, Technical report of the CUIDADO project, 2004.

    :param mag: (..., n_fft, num_frames) Matrix of the frame-by-frame magnitude;
    :param threshold: Ratio of the signal energy to use. Defaults to 0.95;
    :param freq: array of the frequency in Hertz of each frequency bin. If None, result is given as a bin number
    unless rate is set;
    :param rate: Sampling rate of the original signal. If rate is not None, it is used to create freq.
    :return rolloff: (..., 1, num_frames) matrix of the spectral roll-off for each frame, in Hz or #bin.
    """
    energy = torch.square(mag)
    batch_size, fft_size, num_frames = mag.shape
    tot_energy = torch.sum(energy, dim=1, keepdim=True)  # (batch_size, 1, num_frames)
    cumul_energy = torch.cumsum(energy, dim=1)  # (batch_size, fft_size, num_frames)
    transition = torch.where(cumul_energy >= threshold * tot_energy, 1, 0)
    cumul_transition = torch.cumsum(transition, dim=1)  # (batch_size, fft_size, num_frames)
    # indices = (cumul_transition == 1).nonzero(as_tuple=True)
    indices = (cumul_transition - 1) == 0
    # roll_off[indices[0], :, indices[2]] = torch.clone(indices[1]).detach()[:, None].float()
    freq = torch.linspace(0, rate / 2, mag.shape[1])
    # roll_off[indices[0], :, indices[2]] = freq[indices[1]][:, None]
    freq = freq[None, :, None].expand(mag.shape[0], -1, mag.shape[-1])
    # roll_off = torch.zeros_like(tot_energy)
    roll_off = freq[indices]
    roll_off = torch.reshape(roll_off, (mag.shape[0], 1, mag.shape[-1]))
    return roll_off


def spectral_slope(mag: torch.Tensor,
                   rate: int = 1):
    """
    The spectral slope represents the amount of decreasing of the spectral amplitude [1].

    :param mag: (..., N_fft, num_frames) matrix of the frame_by_frame magnitude;
    :param freq: array of the frequency in Hertz of each frequency bin. If None, result is given in bins
    unless rate is set;
    :param rate: Sampling rate of the original signal. If rate is not None, it is used to create freq;
    :return slope: (..., 1, num_frames) array of the frame-wise spectral slope, in Hz or bins depending on freq.
    """
    batch_size, fft_size, num_frames = mag.shape
    freq = torch.linspace(0, rate / 2, fft_size)
    num = fft_size * torch.sum(torch.mul(freq[:, None], mag), dim=1, keepdim=True) - torch.sum(freq) * torch.sum(mag, dim=1, keepdim=True)
    denom = fft_size * torch.sum(torch.square(freq)) - torch.sum(freq) ** 2
    return num / denom


def spectral_flatness(mag, bands: int = 1, rate: int = 1):
    """
    The spectral flatness is a measure of the noisiness of a spectrum. [1, 2]

    :param mag: (..., N_fft, num_frames) matrix of the frame_by_frame magnitude;
    :param bands: Default=1. If int: number of frequency bands to consider, regularly spaced in the spectrum.
                             If list: List of (start, end) for each frequency band. Should be in Hz if rate is set,
                             # bins otherwise;
    :param rate: sampling rate of the signal in Hertz;
    :return flatness: (..., bands, num_frames) matrix of the spectral flatness of each frame and frequency band.
    """
    if mag.ndim >= 3:
        n_fft = mag.shape[-2]
    else:
        n_fft = mag.shape[0]
    out_bands = [(0, n_fft - 1)]
    batch_size, n_fft, num_frames = mag.shape
    flatness = torch.empty((batch_size, len(out_bands), num_frames))
    for (b, band) in enumerate(out_bands):
        arr = mag[:, band[0]:band[1], :]
        flatness[:, b, :] = _geom_mean(arr, dim=1) / torch.mean(arr, dim=1)
    return flatness


def mfcc_torch(audio, rate, num_coeff, mfcc_transform):
    # if transform is None:
    #    transform = torchaudio.transforms.MFCC(sample_rate=rate, n_mfcc=num_coeff)
    mfcc = mfcc_transform(audio)
    means = torch.mean(mfcc, dim=-1)
    maxs = torch.max(mfcc, dim=-1)[0]
    return means, maxs


def get_mfcc(audio, rate, num_coeff):
    mfcc = librosa.feature.mfcc(y=audio, sr=rate)
    return mfcc[:num_coeff]


def phase_fmax_batch(audio, transform=None):
    # TODO: TEST
    """
    Copy of phase_fmax using torch tools for batch processing and GPU compatible processing.
    :param audio:
    :param transform: torchaudio.transforms.Spectrogram. Can be passed to avoid recreating an instance
    at each computation. If None, transform = torchaudio.transforms.Spectrogram(n_fft=2048, hop_length=256)
    :return:
    """
    if transform is None:
        transform = torchaudio.transforms.Spectrogram(n_fft=2048, hop_length=256, power=None)
    stft = transform(audio)[:, 20:256, :]
    mag = torch.abs(stft)
    phase = torch.angle(stft)
    spec_sum = torch.sum(mag, dim=-1)
    max_bins = torch.argmax(spec_sum, dim=-1)
    max_bins_view = max_bins[:, None, None].expand(phase.shape[0], 1, phase.shape[-1])
    phase_freq_max = torch.gather(phase, dim=1, index=max_bins_view)
    phase_freq_max = phase_freq_max[:, 0, :]
    # phase_freq_max = torch.index_select(phase, dim=1, index=max_bins)
    # mag_max_bin_mask = torch.index_select(mag, dim=1, index=max_bins)
    mag_max_bin_mask = torch.gather(mag, dim=1, index=max_bins_view)
    mag_max_bin_mask = mag_max_bin_mask[:, 0, :]
    thresh, _ = torch.max(mag_max_bin_mask, dim=-1, keepdim=True)
    thresh = thresh / 8
    phase_freq_max = torch.where(mag_max_bin_mask > thresh, phase_freq_max, torch.zeros_like(phase_freq_max))
    phase_freq_max_t = phase_freq_max.clone()
    # unwrap phase
    phase_fmax_straight_t = torch.clone(phase_freq_max_t)
    diff_mean_sign = torch.mean(torch.sign(torch.diff(phase_freq_max_t)), dim=-1)
    rolled = torch.roll(phase_freq_max_t, shifts=-1, dims=-1)
    rolled[:, -1] = torch.zeros_like(rolled[:, -1])
    negative = torch.where(phase_freq_max_t - rolled < 0, -1, 0)
    positive = torch.where(phase_freq_max_t - rolled > 0, 1, 0)
    modulos = torch.where(diff_mean_sign[:, None] > 0, positive, negative)
    modulos = torch.cumsum(modulos, dim=-1)
    phase_fmax_straight_t = phase_fmax_straight_t + 2 * torch.pi * modulos
    x_axis_t = torch.arange(0, phase_fmax_straight_t.shape[-1], device=audio.device)
    x_axis_t = torch.vstack([x_axis_t] * audio.shape[0])
    beta_1, beta_0 = util.mean_square_linreg_torch(phase_fmax_straight_t)
    linregerr_t = torch.clone(phase_fmax_straight_t)
    linregerr_t -= (beta_1[:, None] * x_axis_t + beta_0[:, None])
    return linregerr_t


def phase_fmax(sig):
    """
    Copied from:
    https://github.com/henrikjuergens/guitar-fx-extraction/blob/442adab577a090e27de12d779a6d8a0aa917fe1f/featextr.py#L63
    Analyses phase error of frequency bin with maximal amplitude
        compared to pure sine wave"""
    D = librosa.stft(y=sig, hop_length=256)[20:256]
    S, P = librosa.core.magphase(D)
    phase = np.angle(P)
    # plots.phase_spectrogram(phase)

    spec_sum = S.sum(axis=1)
    max_bin = spec_sum.argmax()
    phase_freq_max = phase[max_bin]
    # plots.phase_fmax(phase_freq_max)

    S_max_bin_mask = S[max_bin]
    thresh = S[max_bin].max() / 8
    phase_freq_max = np.where(S_max_bin_mask > thresh, phase_freq_max, 0)
    phase_freq_max_t = np.trim_zeros(phase_freq_max)  # Using only phase with strong signal

    # unwrap phase
    phase_fmax_straight_t = np.copy(phase_freq_max_t)
    diff_mean_sign = np.mean(np.sign(np.diff(phase_freq_max_t)))
    if diff_mean_sign > 0:
        for i in range(1, len(phase_fmax_straight_t)):
            if np.sign(phase_freq_max_t[i - 1]) > np.sign(phase_freq_max_t[i]):
                phase_fmax_straight_t[i:] += 2 * np.pi
    else:
        for i in range(1, len(phase_fmax_straight_t)):
            if np.sign(phase_freq_max_t[i - 1]) < np.sign(phase_freq_max_t[i]):
                phase_fmax_straight_t[i:] -= 2 * np.pi

    x_axis_t = np.arange(0, len(phase_fmax_straight_t))
    coeff = np.polyfit(x_axis_t, phase_fmax_straight_t, 1)
    linregerr_t = np.copy(phase_fmax_straight_t)
    linregerr_t -= (coeff[0] * x_axis_t + coeff[1])
    linregerr_t = np.reshape(linregerr_t, (1, len(linregerr_t)))
    # plots.phase_error_unwrapped(phase_fmax_straight_t, coeff, x_axis_t)
    return linregerr_t


def pitch_curve(audio, rate: int):
    """
    fmin and fmax need to be quite far from one another for the algorithm to work
    :param audio:
    :param rate:
    :param fmin:
    :param fmax:
    :return:
    """
    f0 = torchaudio.functional.detect_pitch_frequency(audio, rate)
    return f0


def rms_energy(audio, torch_compat: bool = False):
    if torch_compat:
        # TODO: make util function for better frame splitting
        # discard last frame that is shorter for now
        frames = torch.stack(torch.split(audio, 1024, dim=-1)[:-1], dim=1)
        return torch.sqrt(torch.mean(torch.square(frames), dim=-1))
    else:
        return librosa.feature.rms(y=audio)
