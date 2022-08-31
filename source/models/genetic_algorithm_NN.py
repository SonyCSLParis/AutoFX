"""
Building a CNN using Genetic Algorithm for training
"""
import pathlib

import pedalboard as pdb
from math import floor
from typing import Tuple

import pygad as ga
import pygad.torchga
import torch
import numpy as np
import torchaudio
from pygad.torchga import torchga
from torch import nn

from mbfx_dataset import MBFXDataset
from multiband_fx import MultiBandFX


class AutoFxModel(nn.Module):
    def _shape_after_conv(self, x: torch.Tensor or Tuple):
        """
        Return shape after Conv2D according to https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        :param x:
        :return:
        """
        if isinstance(x, torch.Tensor):
            batch_size, c_in, h_in, w_in = x.shape
        else:
            batch_size, c_in, h_in, w_in = x
        for seq in self.conv:
            conv = seq[0]
            h_out = floor(
                (h_in + 2 * conv.padding[0] - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1) / conv.stride[0] + 1)
            w_out = floor(
                (w_in + 2 * conv.padding[1] - conv.dilation[1] * (conv.kernel_size[1] - 1) - 1) / conv.stride[1] + 1)
            h_in = h_out
            w_in = w_out
        c_out = self.conv[-1][0].out_channels
        return batch_size, c_out, h_out, w_out

    def __init__(self, fx, num_bands: int, rate: int = 22050, file_size: int = 44100,
                 conv_ch: list[int] or int = 64, conv_k: list[int] or int = 5,
                 conv_stride: list[int] or int = 2, num_conv: int = None,
                 fft_size: int = 1024, hop_size: int = None, spectro_power: int = 2,
                 mel_spectro: bool = True, mel_num_bands: int = 128,
                 learning_rate: float = 0.001):
        super(AutoFxModel, self).__init__()
        if num_conv is None and isinstance(conv_ch, int) and isinstance(conv_k, int) and isinstance(conv_stride, int):
            raise TypeError("num_conv cannot be None if conv_ch, conv_k and conv_stride are int.")
        if num_conv is not None:
            self.num_conv = num_conv
        if isinstance(conv_ch, list):
            self.num_conv = len(conv_ch)
        if isinstance(conv_ch, int):
            conv_ch = [conv_ch] * self.num_conv
        if isinstance(conv_k, int):
            conv_k = [conv_k] * self.num_conv
        if isinstance(conv_stride, int):
            conv_stride = [conv_stride] * self.num_conv
        if hop_size is None:
            hop_size = fft_size // 4
        # TODO: faire toutes les vérifs d'instances pour une utilisation robuste
        self.rate = rate
        self.mbfx = MultiBandFX(fx, num_bands, device=torch.device('cpu'))
        self.num_params = num_bands * len(self.mbfx.settings[0])
        self.conv = nn.ModuleList([])
        for c in range(self.num_conv):
            if c == 0:
                self.conv.append(nn.Sequential(nn.Conv2d(1, conv_ch[c], conv_k[c],
                                                         padding=int(conv_k[c] / 2), stride=conv_stride[c]),
                                               nn.Dropout(),
                                               nn.BatchNorm2d(conv_ch[c]), nn.ReLU()))
            else:
                self.conv.append(nn.Sequential(nn.Conv2d(conv_ch[c - 1], conv_ch[c], conv_k[c],
                                                         stride=conv_stride[c], padding=int(conv_k[c] / 2)),
                                               nn.Dropout(),
                                               nn.BatchNorm2d(conv_ch[c]), nn.ReLU()))
        self.activation = nn.Sigmoid()
        self.loss = nn.L1Loss()
        if mel_spectro:
            self.spectro = torchaudio.transforms.MelSpectrogram(n_fft=fft_size, hop_length=hop_size,
                                                                sample_rate=self.rate,
                                                                power=spectro_power, n_mels=mel_num_bands)
            _, c_out, h_out, w_out = self._shape_after_conv(torch.empty(1, 1, mel_num_bands, file_size // hop_size))
        else:
            self.spectro = torchaudio.transforms.Spectrogram(n_fft=fft_size, hop_length=hop_size, power=spectro_power)
            _, c_out, h_out, w_out = self._shape_after_conv(torch.empty(1, 1, fft_size // 2 + 1, file_size // hop_size))
        self.fcl = nn.Linear(c_out * h_out * w_out, self.num_params)
        self.model = nn.Sequential(self.conv,
                                   self.fcl)


def fitness_func(solution, sol_idx):
    global data_inputs, data_outputs, torch_ga, model, loss_function
    predictions = pygad.torchga.predict(model=model,
                                        solution=solution,
                                        data=data_inputs)
    predictions = model.activation(predictions)
    abs_error = loss_function(predictions, data_outputs).detach().numpy()
    solution_fitness = 1.0 / abs_error
    return solution_fitness


def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))


if __name__ == '__main__':
    import torch
    import pygad

    # Create the PyTorch model.
    model = AutoFxModel([pdb.Distortion, pdb.Gain], 4, num_conv=5)

    # Create an instance of the pygad.torchga.TorchGA class to build the initial population.
    torch_ga = torchga.TorchGA(model=model.model,
                               num_solutions=10)

    loss_function = model.loss

    # Data
    DATASET_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_dry_22050")
    PROCESSED_PATH = pathlib.Path("/home/alexandre/dataset/mbfx_disto_guitar_mono_int")
    dataset = MBFXDataset(PROCESSED_PATH / 'params.csv', DATASET_PATH, PROCESSED_PATH, rate=22050)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=256)
    _, data_inputs, data_outputs = next(iter(dataloader))
    data_inputs = model.spectro(data_inputs)

    # Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
    num_generations = 250  # Number of generations.
    num_parents_mating = 5  # Number of solutions to be selected as parents in the mating pool.
    initial_population = torch_ga.population_weights  # Initial population of network weights

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           initial_population=initial_population,
                           fitness_func=fitness_func,
                           on_generation=callback_generation)
    ga_instance.run()

    # After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
    ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)

    # Returning the details of the best solution.
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

    # Make predictions based on the best solution.
    predictions = pygad.torchga.predict(model=model,
                                        solution=solution,
                                        data=data_inputs)
    print("Predictions : \n", predictions.detach().numpy())

    abs_error = loss_function(predictions, data_outputs)
    print("Absolute Error : ", abs_error.detach().numpy())
