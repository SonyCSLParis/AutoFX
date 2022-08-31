import torch
from torch import nn


class FilmLayer(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(FilmLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear1 = nn.Linear(self.input_size, self.output_size)
        self.linear2 = nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        gamma = self.linear1(x)
        beta = self.linear2(x)
        return gamma, beta
