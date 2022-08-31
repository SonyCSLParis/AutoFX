"""
Function to define a near-perfect-reconstruction pseudo-QMF bank according to
Truong Q Nguyen, Near-Perfect-Reconstruction Pseudo-QMF banks, 1994
"""
import warnings

from matplotlib import pyplot as plt
from scipy import signal
import numpy as np

REMEZ_MAX_ITER = 100


class PseudoQmfBank():
    @staticmethod
    def _error_function(proto, num_bands: int, weights: list[float] = None, num_points: int = 2**15):
        _, h = signal.freqz(proto, [1], worN=num_points)
        if weights is None:
            weights = [1, 1]
        w_step = np.pi / num_points
        h_shifted = np.roll(h, int(-((np.pi / num_bands) / w_step)))
        err = np.max(weights[0] * np.square(h) + weights[1] * np.square(h_shifted) - 1)
        return err

    @staticmethod
    def design_prototype(num_bands: int, tol: float = 1e-16, width: float = None, order: int = None,
                         max_count: int = 10000, init_step_size: float = None, weights=None):
        """
        Optimization process from Creusere & Mitra, 1995
        :param max_count:
        :param init_step_size:
        :param num_bands:
        :param tol:
        :param width:
        :param order:
        :return:
        """
        if weights is None:
            weights = [10, 1]
        if order is None:
            order = int(2 ** (np.log2(num_bands) + 4))
        if width is None:
            width = 1 / (4 * num_bands)
        if init_step_size is None:
            init_step_size = width / 4
        ws = 1 / (2 * num_bands)
        wp = ws - width
        proto = signal.remez(order, [0, wp, ws, 0.5], [1, 0], fs=1, maxiter=REMEZ_MAX_ITER)
        cnt = 0
        step = init_step_size
        dir = 1
        err_old = PseudoQmfBank._error_function(proto, num_bands, weights)
        wp = wp + dir * step
        proto = signal.remez(order, [0, wp, ws, 0.5], [1, 0], fs=1, maxiter=REMEZ_MAX_ITER)
        err = PseudoQmfBank._error_function(proto, num_bands, weights)
        while np.abs(err - err_old) > tol and cnt < max_count:
            if err > err_old:
                dir = -dir
                step /= 2
            wp = wp + dir * step
            proto = signal.remez(order, [0, wp, ws, 0.5], [1, 0], fs=1, maxiter=REMEZ_MAX_ITER)
            err_old = err
            err = PseudoQmfBank._error_function(proto, num_bands, weights)
            cnt += 1
        if cnt == max_count:
            warnings.warn("Max number of iterations reached, the algorithm might not have converged yet.", Warning)
            print("Error was:", err)
        return proto

    @staticmethod
    def show_filter_bank(bank: np.ndarray, title: str, show: bool = False):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for (b, filt) in enumerate(bank):
            w, h = signal.freqz(filt, [1], worN=2000)
            ax.plot(0.5 * w / np.pi, 20 * np.log10(np.abs(h)), label=str(b))
        ax.set_ylim(-100, 5)
        ax.set_xlim(0, 0.5)
        ax.grid(True)
        ax.legend()
        ax.set_xlabel('Normalized frequency')
        ax.set_ylabel('Gain (dB)')
        ax.set_title(title)
        if show:
            plt.show(block=True)
            return None
        else:
            return fig, ax

    def __init__(self, num_bands: int):
        self.prototype = PseudoQmfBank.design_prototype(num_bands)
        self.num_bands = num_bands
        self.analysis_bank = self._init_analysis_bank()
        self.synthesis_bank = self._init_synthesis_bank()

    def analyse(self, sig: np.ndarray):
        if sig.ndim == 1:
            sig = sig[np.newaxis, :]
        out = np.empty((self.num_bands, sig.shape[1]))
        for (b, filt) in enumerate(self.analysis_bank):
            filt_signal = signal.lfilter(filt, [1], sig)
            out[b] = filt_signal
        return out

    def synthesize(self, sig):
        out = signal.lfilter(self.synthesis_bank[0], [1], sig[0])
        for (b, filt) in enumerate(self.synthesis_bank[1:]):
            out += signal.lfilter(filt, [1], sig[b])
        return out

    def anasynth_pipeline(self, sig):
        return self.synthesize(self.analyse(sig))

    def _init_analysis_bank(self, flat: bool = True, zero_endpoints: bool = False):
        ana_bank = np.empty((self.num_bands, len(self.prototype)))
        if flat:
            phase = np.pi / 4
        elif zero_endpoints:    # TODO: Weird results
            phase = np.pi / 2
        for b in range(self.num_bands):
            filt = np.zeros_like(self.prototype)
            for n in range(len(self.prototype)):
                filt[n] = 2 * self.prototype[n] * np.cos((np.pi * (2 * b + 1) * (2 * n + 1) / (4 * self.num_bands)) +
                                                         ((-1) ** b) * phase)
            ana_bank[b, :] = filt
        return ana_bank

    def _init_synthesis_bank(self):
        synth_bank = np.empty((self.num_bands, len(self.prototype)))
        a_even = 0.707
        for b in range(self.num_bands):
            filt = np.zeros_like(self.prototype)
            for n in range(len(self.prototype)):
                filt[n] = 2 * self.prototype[n] * np.cos((2 * b + 1) * np.pi / (2 * self.num_bands) *
                                                         (n - ((len(self.prototype) - 1) / 2)) -
                                                         (-1) ** b * np.pi / 4)
            synth_bank[b, :] = filt
        return synth_bank

    def show_analysis_bank(self, show: bool = True):
        if show:
            PseudoQmfBank.show_filter_bank(self.analysis_bank, "Analysis filters bank", show=show)
            return None
        else:
            fig, ax = PseudoQmfBank.show_filter_bank(self.analysis_bank, "Analysis filters bank", show=show)
            return fig, ax

    def show_synthesis_bank(self, show: bool = True):
        if show:
            PseudoQmfBank.show_filter_bank(self.synthesis_bank, "Synthesis filters bank", show=show)
            return None
        else:
            fig, ax = PseudoQmfBank.show_filter_bank(self.synthesis_bank, "Synthesis filters bank", show=show)
            return fig, ax

    def show_prototype_filter(self, show: bool = True):
        """
        :return:
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        w, h = signal.freqz(self.prototype, [1], worN=2000)
        ax.plot(0.5 * w / np.pi, 20 * np.log10(np.abs(h)))
        ax.set_ylim(-100, 5)
        ax.set_xlim(0, 0.5)
        ax.grid(True)
        ax.set_xlabel('Normalized frequency')
        ax.set_ylabel('Gain (dB)')
        if show:
            plt.show(block=True)
            return None
        else:
            return fig, ax

    def show_overall_frequency_response_analysis(self, show: bool = True):
        """
        :return:
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        resp = np.zeros(2000, dtype=complex)
        for (b, filt) in enumerate(self.analysis_bank):
            w, h = signal.freqz(filt, [1], worN=2000)
            resp += h
        ax.plot(0.5 * w / np.pi, 20 * np.log10(np.abs(resp)))
        # ax.set_ylim(-100, 5)
        # ax.set_xlim(0, 0.5)
        ax.grid(True)
        ax.set_xlabel('Normalized frequency')
        ax.set_ylabel('Gain (dB)')
        if show:
            plt.show(block=True)
            return None
        else:
            return fig, ax
