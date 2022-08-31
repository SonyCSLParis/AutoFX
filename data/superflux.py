"""
Pytorch implementation of
https://github.com/CPJKU/SuperFlux/blob/master/SuperFlux.py
"""
import math
import warnings

import numpy as np
import torch, torchaudio


def maximum_filter(input, size, mode: str = 'constant', origin: list = [0, 0]):
    """
    Apply maximum filtering with kernels of size `size` like
    scipy.ndimage.maximum_filter

    :param input:
    :param size: tuple representing the size of the filter kernels.
    :return:
    """
    if mode != 'constant':
        raise NotImplementedError("Only constant zero padding is available now")
    tensor_size = torch.tensor(size)
    origin = torch.tensor(origin)
    pad_length_start = origin + tensor_size // 2
    pad_length_end = tensor_size // 2 - origin
    # Prepare tensor
    padded_input = torch.zeros((input.shape[0], input.shape[1] + pad_length_start[0] + pad_length_end[0],
                                input.shape[2] + pad_length_start[1] + pad_length_end[1]), device=input.device)
    # fill values
    if pad_length_start[0] == 0 and pad_length_end[0] == 0:
        padded_input[:, :, pad_length_start[1]:-pad_length_end[1]] = input
    elif pad_length_start[1] == 0 and pad_length_end[1] == 0:
        padded_input[:, pad_length_start[0]:-pad_length_end[0], :] = input
    else:
        padded_input[:, pad_length_start[0]:-pad_length_end[0],
        pad_length_start[1]:-pad_length_end[1]] = input
    max_input = torch.nn.functional.max_pool2d(padded_input, size, stride=1)
    return max_input


def maximum_filter1d(input, size, mode, origin):
    """
    Apply 1d max filtering like scipy.ndimage.maximum_filter1d
    :param input:
    :param size:
    :param mode:
    :param origin:
    :return:
    """
    if mode != 'constant':
        raise NotImplementedError("Only constant zero padding is available now")
    pad_length_start = origin + size // 2
    pad_length_end = size // 2 - origin
    # Prepare tensor
    padded_input = torch.zeros((input.shape[0], input.shape[1] + pad_length_start + pad_length_end),
                               device=input.device)
    # fill values
    if pad_length_end == 0:
        padded_input[:, pad_length_start:] = input
    else:
        padded_input[:, pad_length_start:-pad_length_end] = input
    output = torch.nn.functional.max_pool1d(padded_input, size, stride=1)
    return output


def uniform_filter1d(input, size, mode, origin):
    """
    Apply 1d uniform (average) filtering like scipy.ndimage.uniform_filter1d
    :param input:
    :param size:
    :param mode:
    :param origin:
    :return:
    """
    if mode != 'constant':
        raise NotImplementedError("Only constant zero padding is available now")
    pad_length_start = origin + size // 2
    pad_length_end = size // 2 - origin
    # Prepare tensor
    padded_input = torch.zeros((input.shape[0], input.shape[1] + pad_length_start + pad_length_end),
                               device=input.device)
    # fill values
    if pad_length_end == 0:
        padded_input[:, pad_length_start:] = input
    else:
        padded_input[:, pad_length_start:-pad_length_end] = input
    output = torch.nn.functional.avg_pool1d(padded_input, size, stride=1)
    return output


class Filter(object):
    @staticmethod
    def frequencies(bands, fmin, fmax, a=440):
        """
        Returns a list of frequencies aligned on a logarithmic scale.
        :param bands: number of filter bands per octave
        :param fmin:  the minimum frequency [Hz]
        :param fmax:  the maximum frequency [Hz]
        :param a:     frequency of A0 [Hz]
        :returns:     a list of frequencies
        Using 12 bands per octave and a=440 corresponding to the MIDI notes.
        """
        # factor 2 frequencies are apart
        factor = 2.0 ** (1.0 / bands)
        # start with A0
        freq = a
        frequencies = [freq]
        # go upwards till fmax
        while freq <= fmax:
            # multiply once more, since the included frequency is a frequency
            # which is only used as the right corner of a (triangular) filter
            freq *= factor
            frequencies.append(freq)
        # restart with a and go downwards till fmin
        freq = a
        while freq >= fmin:
            # divide once more, since the included frequency is a frequency
            # which is only used as the left corner of a (triangular) filter
            freq /= factor
            frequencies.append(freq)
        # sort frequencies
        frequencies.sort()
        # return the list
        return frequencies

    @staticmethod
    def triangular_filter(start, mid, stop, equal: bool = False):
        """
        Calculates a triangular filter of the given size.
        :param start: start bin (with value 0, included in the filter)
        :param mid:   center bin (of height 1, unless norm is True)
        :param stop:  end bin (with value 0, not included in the filter)
        :param equal: normalize the area of the filter to 1
        :returns:     a triangular shaped filter
        """
        height = 1
        if equal:
            height = 2 / (stop - start)
        triangular_filter = torch.empty((int(stop - start),))
        rising = torch.linspace(0, height - int(height / (mid - start)),
                                int(mid - start))
        triangular_filter[:int(mid - start)] = rising
        falling = torch.linspace(height, 0 + int(height / (stop - mid)),
                                 int(stop - mid))
        triangular_filter[int(mid - start):] = falling
        return triangular_filter

    def __init__(self, num_fft_bins, rate, bands: int = 24,
                 fmin: float = 30, fmax: float = 17000, equal: bool = False):
        """
        Creates a new Filter object instance.
        :param num_fft_bins: number of FFT coefficients
        :param rate:           sample rate of the audio file
        :param bands:        number of filter bands
        :param fmin:         the minimum frequency [Hz]
        :param fmax:         the maximum frequency [Hz]
        :param equal:        normalize the area of each band to 1
        """
        self.rate = rate
        fmax = rate / 2 if fmax > rate / 2 else fmax
        frequencies = self.frequencies(bands, fmin, fmax)
        # print(frequencies)
        # conversion factor for mapping of frequencies to spectrogram bins
        factor = (rate / 2.0) / num_fft_bins
        # map frequencies to spectro bins
        frequencies = torch.round(torch.tensor(frequencies) / factor)
        frequencies = torch.unique(frequencies)
        # filter out all frequencies outside valid range
        frequencies = [f for f in frequencies if f < num_fft_bins]
        bands = len(frequencies) - 2
        self.filterbank = torch.zeros((num_fft_bins, bands), dtype=torch.float)
        for band in range(bands):
            start, mid, stop = frequencies[band:band + 3]
            triangular_filter = self.triangular_filter(start, mid, stop, equal)
            self.filterbank[int(start):int(stop), band] = triangular_filter


class Spectrogram(object):
    """
    Spectrogram class
    """

    def __init__(self, audio, rate, frame_size=2048, fps=200, filterbank=None,
                 log=False, mul=1, add=1, online=True, block_size=2048,
                 lgd=False):
        """
        Creates a new Spectrogram object instance and performs a STFT on the
        given audio.
        :param audio:        audio file
        :param rate:        sampling rate of audio
        :param frame_size: the size for the window [samples]
        :param fps:        frames per second
        :param filterbank: use the given filterbank for dimensionality
                           reduction
        :param log:        use logarithmic magnitude
        :param mul:        multiply the magnitude by this factor before taking
                           the logarithm
        :param add:        add this value to the magnitude before taking the
                           logarithm
        :param online:     work in online mode (i.e. use only past information)
        :param block_size: perform the filtering in blocks of the given size
        :param lgd:        compute the local group delay (needed for the
                           ComplexFlux algorithm)
        """
        self.audio = audio
        self.rate = rate
        self.fps = fps
        self.filterbank = filterbank
        if add <= 0:
            raise ValueError("a positive value must be added before taking "
                             "the logarithm")
        if mul <= 0:
            raise ValueError("a positive value must be multiplied before "
                             "taking the logarithm")
        self.hop_size = rate // fps
        self.num_frames = audio.shape[-1] // self.hop_size
        self.num_fft_bins = int(frame_size / 2) + 1
        self.num_bins = self.num_fft_bins
        if filterbank is None:
            self.spec = torch.empty((self.num_frames, self.num_fft_bins), dtype=torch.float32)
        else:
            self.spec = torch.empty((self.num_frames, filterbank.shape[1]), dtype=torch.float32)
            self.num_bins = filterbank.shape[1]
        if not block_size or block_size > self.num_frames:
            block_size = self.num_frames
        self.lgd = None
        if lgd:
            warnings.warn("Local group delay not implemented yet.", UserWarning)
        self.window = torch.hann_window(frame_size)
        transform = torchaudio.transforms.Spectrogram(frame_size, frame_size, self.hop_size, power=1,
                                                      window_fn=lambda size: torch.hann_window(size, device=audio.device))
        stft = transform(audio)
        stft = torch.abs(stft)
        if filterbank is None:
            self.spec = stft
        else:
            self.spec = torch.matmul(torch.transpose(stft, 1, 2), filterbank)
        if log:
            self.spec = torch.log10(mul * self.spec + add)


class SpectralODF(object):
    """
    The SpectralODF class implements most of the common onset detection
    function based on the magnitude or phase information of a spectrogram.
    """

    def __init__(self, spectrogram, ratio=0.5, max_bins=3, diff_frames=None,
                 temporal_filter=3, temporal_origin=0):
        """
        Creates a new ODF object instance.
        :param spectrogram:     a Spectrogram object on which the detection
                                functions operate
        :param ratio:           calculate the difference to the frame which
                                has the given magnitude ratio
        :param max_bins:        number of bins for the maximum filter
        :param diff_frames:     calculate the difference to the N-th previous
                                frame
        :param temporal_filter: temporal maximum filtering of the local group
                                delay for the ComplexFlux algorithms
        :param temporal_origin: origin of the temporal maximum filter
        If no diff_frames are given, they are calculated automatically based on
        the given ratio.
        """
        self.s = spectrogram
        # determine the number off diff frames
        if diff_frames is None:
            # get the first sample with a higher magnitude than given ratio
            sample = torch.argmax(torch.where(self.s.window > ratio, 1, 0))
            diff_samples = len(self.s.window) / 2 - sample
            # convert to frames
            diff_frames = int(diff_samples / self.s.hop_size)
            # set the minimum to 1
            if diff_frames < 1:
                diff_frames = 1
        self.diff_frames = diff_frames
        # number of bins used for the maximum filter
        self.max_bins = max_bins
        self.temporal_filter = temporal_filter
        self.temporal_origin = temporal_origin

    @staticmethod
    def _superflux_diff_spec(spec, diff_frames=1, max_bins=3):
        """
        Calculate the difference spec used for SuperFlux.
        :param spec:        magnitude spectrogram
        :param diff_frames: calculate the difference to the N-th previous frame
        :param max_bins:    number of neighboring bins used for maximum
                            filtering
        :return:            difference spectrogram used for SuperFlux
        Note: If 'max_bins' is greater than 0, a maximum filter of this size
              is applied in the frequency direction. The difference of the
              k-th frequency bin of the magnitude spectrogram is then
              calculated relative to the maximum over m bins of the N-th
              previous frame (e.g. m=3: k-1, k, k+1).
              This method works only properly if the number of bands for the
              filterbank is chosen carefully. A values of 24 (i.e. quarter-tone
              resolution) usually yields good results.
        """
        # init diff matrix
        diff_spec = torch.zeros_like(spec)
        if diff_frames < 1:
            raise ValueError("number of diff_frames must be >= 1")
        # widen the spectrogram in frequency dimension by `max_bins`
        if spec.ndim == 2:
            spec = spec[None, :, :]
        max_spec = maximum_filter(spec, size=[1, max_bins])
        # calculate the diff
        diff_spec[:, diff_frames:] = spec[:, diff_frames:] - max_spec[:, 0:-diff_frames]
        # keep only positive values
        diff_spec = torch.nn.functional.relu(diff_spec)
        # return diff spec
        return diff_spec

    # Onset Detection Functions
    def superflux(self):
        """
        SuperFlux with a maximum filter based vibrato suppression.
        :return: SuperFlux onset detection function
        "Maximum Filter Vibrato Suppression for Onset Detection"
        Sebastian Böck and Gerhard Widmer.
        Proceedings of the 16th International Conference on Digital Audio
        Effects (DAFx-13), Maynooth, Ireland, September 2013
        """
        # compute the difference spectrogram as in the SuperFlux algorithm
        diff_spec = self._superflux_diff_spec(self.s.spec, self.diff_frames,
                                              self.max_bins)
        # sum all positive 1st order max. filtered differences
        return torch.sum(diff_spec, axis=1)


class Onset(object):
    """
    Onset Class.
    """

    def __init__(self, activations, fps, online=False, sep=''):
        """
        Creates a new Onset object instance with the given activations of the
        ODF (OnsetDetectionFunction). The activations can be read from a file.
        :param activations: an array containing the activations of the ODF
        :param fps:         frame rate of the activations
        :param online:      work in online mode (i.e. use only past
                            information)
        """
        self.activations = None  # activations of the ODF
        self.fps = fps  # frame rate of the activation function
        self.online = online  # online peak-picking
        self.detections = []  # list of detected onsets (in seconds)
        self.detect_activations = []
        # set / load activations
        if isinstance(activations, torch.Tensor):
            # activations are given as an array
            self.activations = activations
        else:
            # read in the activations from a file
            self.load(activations, sep)

    def detect(self, threshold, combine=30, pre_avg=0.15, pre_max=0.01,
               post_avg=0, post_max=0.05, delay=0, num_onset=None):
        """
        Detects the onsets.
        :param threshold: threshold for peak-picking
        :param combine:   only report 1 onset for N seconds
        :param pre_avg:   use N seconds past information for moving average
        :param pre_max:   use N seconds past information for moving maximum
        :param post_avg:  use N seconds future information for moving average
        :param post_max:  use N seconds future information for moving maximum
        :param delay:     report the onset N seconds delayed
        In online mode, post_avg and post_max are set to 0.
        Implements the peak-picking method described in:
        "Evaluating the Online Capabilities of Onset Detection Methods"
        Sebastian Böck, Florian Krebs and Markus Schedl
        Proceedings of the 13th International Society for Music Information
        Retrieval Conference (ISMIR), 2012
        """
        # online mode?
        if self.online:
            post_max = 0
            post_avg = 0
        # convert timing information to frames
        pre_avg = int(round(self.fps * pre_avg))
        pre_max = int(round(self.fps * pre_max))
        post_max = int(round(self.fps * post_max))
        post_avg = int(round(self.fps * post_avg))
        # convert to seconds
        combine /= 1000.
        delay /= 1000.
        # init detections
        self.detections = []
        # moving maximum
        max_length = pre_max + post_max + 1
        max_origin = int(math.floor((pre_max - post_max) / 2))
        if self.activations.ndim == 1:
            self.activations = self.activations[None, :]
        mov_max = maximum_filter1d(self.activations, max_length,
                                   mode='constant', origin=max_origin)
        # moving average
        avg_length = pre_avg + post_avg + 1
        avg_origin = int(math.floor((pre_avg - post_avg) / 2))
        mov_avg = uniform_filter1d(self.activations, avg_length,
                                   mode='constant', origin=avg_origin)
        # detections are activation equal to the moving maximum
        detections = self.activations * (self.activations == mov_max)
        # detections must be greater or equal than the mov. average + threshold
        detections *= (detections >= mov_avg + threshold)
        # onset_activations = self.activations[torch.nonzero(detections, as_tuple=True)]
        onset_activations = torch.where(detections != 0, self.activations, torch.scalar_tensor(0, device=self.activations.device))
        # onset_activations, sorting_indices = torch.sort(onset_activations, dim=1, descending=True, stable=True)
        # if num_onset is not None:
        #     print("YOOOO", num_onset)
        #     onset_activations = onset_activations[:, :num_onset])
        # print("WASSUP", sorting_indices)
        # onset_activations = onset_activations[:, 0]
        # detections = torch.nonzero(detections[0]) / self.fps
        # detections = detections[:, 0]
        # convert detected onsets to a list of timestamps
        detections = torch.where(detections != 0, detections / self.fps, torch.roll(detections, 1, dims=1) / self.fps)
        # sort detections like onset_activations
        # tmp = torch.zeros_like(detections)
        # tmp = tmp.scatter_(1, sorting_indices, detections)
        # detections = tmp.clone()
        # if num_onset is not None:
        #    detections = detections[:, :num_onset]
        # shift if necessary
        if delay != 0:
            detections += delay
        # always use the first detection and all others if none was reported
        # within the last `combine` seconds
        if len(detections[0]) > 1:
            # filter all detections which occur within `combine` seconds
            # combined_detections = detections[:, 1:][torch.diff(detections) > combine]
            combined_detections = torch.where(torch.diff(detections) > combine,
                                              detections[:, 1:], torch.scalar_tensor(999, device=detections.device))
            # combined_activations = onset_activations[:, 1:][torch.diff(detections) > combine]
            combined_activations = torch.where(torch.diff(detections) > combine,
                                               onset_activations[:, 1:], torch.scalar_tensor(0, device=onset_activations.device))
            # add them after the first detection
            self.detections = torch.cat([detections[:, 0][:, None], combined_detections], dim=1)
            self.detect_activations = torch.cat([onset_activations[:, 0][:, None], combined_activations],
                                                dim=1)
        else:
            self.detections = detections
            self.detect_activations = onset_activations

    def write(self, filename):
        """
        Write the detected onsets to the given file.
        :param filename: the target file name
        Only useful if detect() was invoked before.
        """
        with open(filename, 'w') as f:
            for pos in self.detections:
                f.write(str(pos) + '\n')

    def save(self, filename, sep):
        """
        Save the onset activations to the given file.
        :param filename: the target file name
        :param sep: separator between activation values
        Note: using an empty separator ('') results in a binary numpy array.
        """
        self.activations.numpy().tofile(filename, sep=sep)

    def load(self, filename, sep):
        """
        Load the onset activations from the given file.
        :param filename: the target file name
        :param sep: separator between activation values
        Note: using an empty separator ('') results in a binary numpy array.
        """
        self.activations = torch.tensor(np.fromfile(filename, sep=sep))
