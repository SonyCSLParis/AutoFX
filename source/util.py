"""
Utility functions for data processing.
[1] Stein et al., Automatic detection of audio effects, AES 2010.
"""
import pathlib
import warnings
from typing import Tuple, Any, List

import librosa.onset
import scipy.fftpack
import pedalboard as pdb
import torch
from pedalboard.io import AudioFile
import soundfile as sf
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from numpy.typing import ArrayLike

NUM_COEFF_CEPSTRUM = 10
GUITAR_MIN_FREQUENCY = 80
GUITAR_MAX_FREQUENCY = 1200
CLASSES = ['Dry', 'Feedback Delay', 'Slapback Delay', 'Reverb', 'Chorus', 'Flanger', 'Phaser',
           'Tremolo', 'Vibrato', 'Distortion', 'Overdrive']


def get_fx_params(fx: pdb.Plugin or List[pdb.Plugin] or pdb.Pedalboard):
    settings = []
    if isinstance(fx, pdb.Plugin) and not isinstance(fx, pdb.Pedalboard):
        fx = [fx]
    for f in fx:
        fx_settings = {}
        items = list(f.__class__.__dict__.items())
        for item in items:
            if isinstance(item[1], property):
                fx_settings[item[0]] = item[1].__get__(f, f.__class__)
        settings.append(fx_settings)
    return settings


def get_params_iirfilter(fx: pdb.Plugin, suffix: str = ''):
    settings = []
    fx_settings = {}
    s = fx.__repr__()
    s = s.split('=')
    if suffix != '':
        suffix = '_' + suffix
    fx_settings['cutoff_frequency_hz' + suffix] = float(s[1].split(' ')[0])
    fx_settings['gain_db' + suffix] = float(s[2].split(' ')[0])
    fx_settings['q' + suffix] = float(s[3].split(' ')[0])
    settings.append(fx_settings)
    return settings

def set_fx_params(fx: pdb.Plugin or List[pdb.Plugin] or pdb.Pedalboard, params: dict or List[dict]):
    if isinstance(fx, List | pdb.Pedalboard) and isinstance(params, List | np.ndarray | torch.Tensor):
        if len(fx) != len(params):
            raise TypeError("Fx Board and Parameters list must have the same length.")
    if not isinstance(fx, List | pdb.Pedalboard):
        fx = [fx]
    if not isinstance(params, List | np.ndarray | torch.Tensor):
        params = [params]
    for i in range(len(params)):
        items = list(fx[i].__class__.__dict__.items())
        for item in items:
            if isinstance(item[1], property):
                if item[0] not in params[i].keys():
                    warnings.warn(f'{item[0]} not found in params. Keeping previous value.', UserWarning)
                else:
                    item[1].__set__(fx[i], params[i][item[0]])
    return fx


def plot_response(fs, w, h, title):
    """Utility function to plot response functions"""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(0.5 * fs * w / np.pi, 20 * np.log10(np.abs(h)))
    ax.set_ylim(-40, 5)
    ax.set_xlim(0, 0.5 * fs)
    ax.grid(True)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Gain (dB)')
    ax.set_title(title)


def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          unbalanced_set=False,
                          title=None):
    """
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
    unbalanced_set: If True, displayed values are normalized per class instead than by the total dataset size.
    title:         Title for the heatmap. Default is None.
    """

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks
    if percent:
        if not unbalanced_set:
            group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
        else:
            group_percentages = ["{0:.2%}".format(value) for value in (cf / np.sum(cf, axis=1)[:, None]).flatten()]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize is None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if not xyticks:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    fig = plt.figure(figsize=figsize)
    if unbalanced_set:
        cf = cf / np.sum(cf, axis=1)[:, None]
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)
    return fig


def apply_fx(audio, rate: float, board: pdb.Pedalboard):
    """
    Apply effects to audio.
    :param audio: Array representing the audio to process;
    :param rate: Sample rate of the audio to process;
    :param board: Pedalboard instance of FX to apply. Can contain one or several FX;
    :return: Processed audio.
    """
    return board.process(audio, rate)


def read_audio(path: str or pathlib.Path, normalize: bool = True, add_noise: bool = False, cut_beginning: float = None,
               **kwargs) -> \
        Tuple[ArrayLike, float]:
    """
    Wrapper function to read an audio file using pedalboard.io
    :param path: Path to audio file to read;
    :param normalize: Should the output file be normalized in loudness. Default is True.
    :param add_noise: add white noise to the signal to avoid division by zero. Default is False.
    :param cut_beginning: time to cut from the sample in seconds.
    :param kwargs: Keyword arguments to pass to soundfile.read;
    :return audio, rate: Read audio and the corresponding sample rate.
    """
    with AudioFile(str(path), 'r') as f:
        audio = f.read(f.frames)
        rate = f.samplerate
    if normalize:
        audio = audio / np.max(np.abs(audio))
    if add_noise:
        audio += np.random.normal(0, 1e-9, len(audio))
    if cut_beginning is not None:
        audio = audio[:, int(cut_beginning * rate):]
    return audio, rate


def energy_envelope(audio: ArrayLike, rate: float,
                    window_size: float = 100, method: str = 'rms') -> Tuple[ArrayLike, ArrayLike]:
    """
    Compute the energy envelope of a signal according to the selected method. A default window size
    of 100ms is used for a 5Hz low-pass filtering (see Peeters' Cuidado Project report, 2003).

    Valid methods are Root Mean Square ('rms') and Absolute ('abs').

    :param audio: Array representing the signal to be analysed;
    :param rate: sampling rate of the signal;
    :param window_size: Window size in ms for low-pass filtering. Default: 100;
    :param method: name of the method to use. Default: 'rms'.
    :return (env, times): energy envelope of the signal and the corresponding times in seconds.
    """
    if method == 'rms':
        window_size_n = window_size * rate / 1000
        chunks = np.ceil(audio.size / window_size_n)
        env = list(map(lambda x: np.sqrt(np.mean(np.square(x))), np.array_split(audio, chunks)))
        times = np.arange(chunks) * window_size / 1000
        return np.array(env), times
    elif method == 'abs':
        window_size_n = window_size * rate / 1000
        chunks = np.ceil(audio.size / window_size_n)
        env = list(map(lambda x: np.mean(np.abs(x)), np.array_split(audio, chunks)))
        times = np.arange(chunks) * window_size / 1000
        return np.array(env), times
    else:
        raise NotImplementedError


def find_attack_torch(energy,
                      start_threshold: float, end_threshold: float):
    """
    Find beginning and end of attack from energy envelope using a fixed method.

    See: Geoffroy Peeters, A large set of audio features for sound description in the CUIDADO project, 2003.


    :param energy: ArrayLike of the energy envelope;
    :param end_threshold: max energy ratio to detect end of attack in 'fixed' method;
    :param start_threshold: max energy ratio to detect start of attack in 'fixed' method;
    :return (start, end): indices from energy to the corresponding instants.
    """
    max_energy, max_pos = torch.max(energy, dim=-1, keepdim=True)
    start = torch.where(energy >= max_energy * start_threshold, 1, 0)
    end = torch.where(energy >= max_energy * end_threshold, 1, 0)
    _, start = torch.max(start, dim=-1, keepdim=True)
    _, end = torch.max(end, dim=-1, keepdim=True)
    return start, end


def find_attack(energy: ArrayLike, method: str,
                start_threshold: float = None, end_threshold: float = None,
                times: ArrayLike = None) -> Tuple[float, float]:
    """
    Find beginning and end of attack from energy envelope using a fixed or adaptive threshold method.

    See: Geoffroy Peeters, A large set of audio features for sound description in the CUIDADO project, 2003.


    :param energy: ArrayLike of the energy envelope;
    :param method: 'adaptive' or 'fixed';
    :param end_threshold: max energy ratio to detect end of attack in 'fixed' method;
    :param start_threshold: max energy ratio to detect start of attack in 'fixed' method;
    :param times: timestamps in seconds corresponding to the energy samples. If set, start and end are given in seconds.
    :return (start, end): indices from energy to the corresponding instants.
    """
    if method == 'fixed':
        if start_threshold is None or end_threshold is None:
            raise ValueError("For 'fixed' method, start and end thresholds have to be set.")
        max_energy = np.max(energy)
        max_pos = np.argmax(energy)
        start, end = None, None
        for i in range(max_pos):
            if energy[i] >= max_energy * start_threshold and start is None:
                start = i
            if energy[i] >= max_energy * end_threshold and end is None:
                end = i
        if times is not None:
            start = times[start]
            end = times[end]
        return start, end
    elif method == 'adaptive':
        raise NotImplementedError("Will be added if necessary")
        # if start_threshold is not None or end_threshold is not None:
        #    raise UserWarning("Setting thresholds is useless in 'adaptive' method. Values are ignored.")
    else:
        raise NotImplementedError("method should be 'fixed' or 'adaptive'")


def get_stft(audio: ArrayLike, rate: float, fft_size: int, hop_size: int = None, window: Any = 'hann',
             window_size: int = None) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Wrapper function to obtain the Short Time Fourier Transform of a sound. As of now, simply calls scipy.signal.

    :param audio: ArrayLike of the sound to analyse;
    :param rate: sampling rate of the audio signal. Necessary to return the correct frequencies;
    :param fft_size: Size of the fft frames in samples. If fft_size > window_size, the windowed signal is zero padded;
    :param hop_size: hop size between frames in samples;
    :param window: window to use, can be a string, function, array...
    :param window_size: size of the window in samples.
    :return stft: Complex-valued matrix of short-term Fourier transform coefficients.
    :return freq: array of the frequency bins values in Hertz.
    """
    if window_size is None:
        window_size = fft_size
    if hop_size is None:
        hop_size = max(window_size // 16, 1)
    freq, times, stft = signal.stft(audio, fs=rate, nfft=fft_size, noverlap=window_size - hop_size,
                                    nperseg=window_size, window=window)
    return stft[0], freq, times


def hi_pass(arr: ArrayLike, method: str = 'simple'):
    """
    Simple High-pass filtering function.

    :param arr: Signal to filter;
    :param method: type of filtering to apply. Default is simply subtracting previous value to the current one.
    :return: High-passed version of input signal
    """
    out = np.zeros_like(arr)
    if method == 'simple':
        out[1:] = arr[1:] - arr[:-1]
        return out
    else:
        return NotImplemented


def derivative(arr: ArrayLike, step: float, method: str = 'newton'):
    """
    Returns the derivative of arr.

    :param arr: Signal to differentiate;
    :param step: step between samples;
    :param method: type of derivation algorithm to use. Default is Newton's difference quotient;
    :return:
    """
    out = np.zeros_like(arr)
    if method == 'newton':
        out[1:] = (arr[:-1] - arr[1:]) / step
        return out
    else:
        return NotImplemented


def mean(arr: ArrayLike):
    """
    Wrapper function to compute the mean value of a signal.

    :param arr: Input signal.
    :return: Mean value of the input signal
    """
    return np.mean(arr)


def std(arr: ArrayLike):
    """
    Wrapper function to compute the standard deviation of a signal.

    :param arr: input signal
    :return: Standard deviation of the input signal
    """
    return np.std(arr)


def get_cepstrum(mag: ArrayLike, full: bool = False, num_coeff: int = NUM_COEFF_CEPSTRUM):
    """
    Obtain cepstrum as explained in [1].

    :param mag: matrix of the frame-by-frame magnitude spectra of the input signal;
    :param full: defines if the function returns the complete cepstrum of simply the first coeff. Default is False.
    :param num_coeff: defines the number of coefficients to keep from the cepstrum.
    :return: First coefficients of the cepstrum of full cepstrum.
    """
    log_sq_mag = np.log(np.square(mag))
    dct = scipy.fftpack.dct(log_sq_mag)
    if full:
        return dct
    return dct[:num_coeff]


def f0_spectral_product(mag: ArrayLike, freq: ArrayLike, rate: float, decim_factor: int,
                        f_min: float = 0.75 * GUITAR_MIN_FREQUENCY, f_max: float = 1.5 * GUITAR_MAX_FREQUENCY,
                        fft_size: int = None) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Obtain the fundamental frequency of a signal using the spectral product technique.


    :param mag: Array of the real spectrum magnitudes;
    :param freq: array of the frequencies in Hertz for the corresponding fft bins;
    :param rate: sampling rate of the signal in Hertz;
    :param decim_factor: number of times the spectrum is decimated;
    :param f_min: Minimum frequency in Hz to look for f0. Default is 0.75*GUITAR_MIN_FREQUENCY;
    :param f_max: Maximum frequency in Hz to look for f0. Default is 1.5*GUITAR_MAX_FREQUENCY;
    :param fft_size: Size of the fft. If None, inferred for freq's shape. Default is None.
    :return (f0, sp_mag, sp_freq):  fundamental frequency and the accompanying magnitude and frequencies
            of the spectral product.
    """
    if fft_size is None:
        fft_size = (len(freq) - 1) * 2
    bin_min = int(f_min / rate * (fft_size / 2 + 1))
    bin_max = int(f_max / rate * (fft_size / 2 + 1))
    sp_max = int(fft_size / (2 * decim_factor))
    if bin_max > sp_max:
        bin_max = sp_max
    sp_mag = np.ones(sp_max)
    for dec in range(1, decim_factor + 1):
        sp_mag *= mag[::dec][:sp_max]
    sp_freq = freq[:sp_max]
    f0 = sp_freq[np.argmax(sp_mag[bin_min:bin_max]) + bin_min]
    return f0, sp_mag, sp_freq


def midi2hertz(midi_pitch: int) -> float:
    return 440 * 2 ** ((midi_pitch - 69) / 12)


def hertz2midi(freq: float) -> int:
    return 69 + 12 * np.log2(freq / 440)


def idmt_fx2one_hot_vector(fx: str) -> np.ndarray:
    vector = np.zeros(11)
    match fx:
        case 'Dry':
            vector[0] = 1
        case 'Amp sim':
            vector[0] = 1
        case 'Feedback delay':
            vector[1] = 1
        case 'Slapback delay':
            vector[2] = 1
        case 'Reverb':
            vector[3] = 1
        case 'Chorus':
            vector[4] = 1
        case 'Flanger':
            vector[5] = 1
        case 'Phaser':
            vector[6] = 1
        case 'Tremolo':
            vector[7] = 1
        case 'Vibrato':
            vector[8] = 1
        case 'Distortion':
            vector[9] = 1
        case 'Overdrive':
            vector[10] = 1
        case _:
            raise ValueError("Unknown FX")
    return vector


def idmt_fx(char: str):
    match char:
        case '11':
            return 'Dry'
        case '12':
            return 'Amp sim'
        case '21':
            return 'Feedback delay'
        case '22':
            return 'Slapback delay'
        case '23':
            return 'Reverb'
        case '31':
            return 'Chorus'
        case '32':
            return 'Flanger'
        case '33':
            return 'Phaser'
        case '34':
            return 'Tremolo'
        case '35':
            return 'Vibrato'
        case '41':
            return 'Distortion'
        case '42':
            return 'Overdrive'
        case _:
            return None

def idmt_fx2class_number(fx: str) -> int:
    match fx:
        case 'Dry':
            return 0
        case 'Amp sim':
            return 0
        case 'Feedback delay':
            return 1
        case 'Slapback delay':
            return 2
        case 'Reverb':
            return 3
        case 'Chorus':
            return 4
        case 'Flanger':
            return 5
        case 'Phaser':
            return 6
        case 'Tremolo':
            return 7
        case 'Vibrato':
            return 8
        case 'Distortion':
            return 9
        case 'Overdrive':
            return 10
        case _:
            raise ValueError("Unknown FX")


def class_number2idmt_fx(cls: int) -> str:
    match cls:
        case 0:
            return 'Dry or Amp sim'
        case 1:
            return 'Feedback delay'
        case 2:
            return 'Slapback delay'
        case 3:
            return 'Reverb'
        case 4:
            return 'Chorus'
        case 5:
            return 'Flanger'
        case 6:
            return 'Phaser'
        case 7:
            return 'Tremolo'
        case 8:
            return 'Vibrato'
        case 9:
            return 'Distortion'
        case 10:
            return 'Overdrive'
        case _:
            raise ValueError("Unknown FX")


def cut2onset(audio, rate, pre_max: int = 20000, post_max: int = 20000, **kwargs):
    onset = librosa.onset.onset_detect(y=audio, sr=rate, units='samples',
                                       post_max=post_max, pre_max=pre_max, **kwargs)
    if len(onset) > 1:
        raise ValueError("Several onsets detected. Aborting.")
    else:
        return audio[onset[0]:]


def mean_square_linreg_torch(tens, x=None):
    """
    Linear regression using ordinary mean squares estimator.
    beta_1 is the slope and beta_0 is the intercept
    :param tens:
    :param x:
    :return:
    """
    batch_size = tens.shape[0]
    length = tens.shape[-1]
    if x is None:
        x = torch.arange(0, length, step=1, dtype=torch.float, device=tens.device)
        x = torch.vstack([x] * batch_size)
    x_mean = torch.mean(x, dim=-1, keepdim=True)
    y_mean = torch.mean(tens, dim=-1, keepdim=True)
    beta_1 = torch.sum((x - x_mean) * (tens - y_mean), dim=-1) / torch.sum(torch.square(x - x_mean), dim=-1)
    beta_0 = y_mean - beta_1[:, None] * x_mean
    return beta_1, beta_0[:, 0]


def str2pdb(fx: str):
    match fx:
        case "chorus":
            return pdb.Chorus
        case "delay":
            return pdb.Delay
        case _:
            raise NotImplementedError


def param_range_from_cli(param_range: list[str]):
    i = 0
    out = []
    while i < len(param_range):
        tup = param_range[i] + ',' + param_range[i+1]
        tup = tup[1:-1]     # drop parentheses
        tup = tuple(map(float, tup.split(',')))
        out.append(tup)
        i += 2
    return out


def approx_argmax(arr, beta: int = 10):
    """
    Differentiable approximation of argmax.
    :param arr: (..., length) array to apply argmax to
    :param beta (Default = 10): scaling parameter. The higher the better the approximation.
    It should not be too high to avoid exceeding capacity.
    :return:
    """
    maximum, _ = torch.max(arr, dim=-1)
    arr = arr - maximum[:, None]
    batch_size, length = arr.shape
    i = torch.arange(length, device=arr.device)
    estim = (torch.sum(i * torch.exp(beta * arr), dim=-1, keepdim=True)) / \
            (torch.sum(torch.exp(beta * arr), dim=-1, keepdim=True))
    return estim


def approx_argmax2(arr, beta: int = 100):
    """
    Differentiable approximation of argmax with log for avoiding NaNs.
    :param arr: (..., length) array to apply argmax to
    :param beta (Default = 100): scaling parameter. The higher the better the approximation.
    It should not be too high to avoid exceeding capacity.
    :return:
    """
    batch_size, length = arr.shape
    i = torch.arange(length, device=arr.device)
    estim = (torch.sum(i * torch.pow(arr, beta), dim=-1, keepdim=True)) / \
            (torch.sum(torch.pow(arr, beta), dim=-1, keepdim=True))
    return estim


def aggregated_class(fx):
    match fx:
        case 0:
            return 5
        case 1:
            return 1
        case 2:
            return 1
        case 3:
            return 3
        case 4:
            return 0
        case 5:
            return 0
        case 6:
            return 0
        case 7:
            return 4
        case 8:
            return 0
        case 9:
            return 2
        case 10:
            return 2
        case _:
            raise ValueError(f"Unknown Fx {fx}")
