import pathlib

import pygad
import numpy as np
import torch
import torchaudio

from mbfx_dataset import MBFXDataset


def fitness_func(solution, sol_idx):
    global data_inputs, data_outputs
    solution = np.reshape(solution, (len(data_inputs), 8))
    predictions = data_inputs @ solution
    abs_error = np.abs(predictions - data_outputs)
    solution_fitness = 1.0 / np.sum(abs_error)
    return solution_fitness


def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))


if __name__ == '__main__':
    # Data
    DATASET_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_dry_22050")
    PROCESSED_PATH = pathlib.Path("/home/alexandre/dataset/mbfx_disto_guitar_mono_int")
    dataset = MBFXDataset(PROCESSED_PATH / 'params.csv', DATASET_PATH, PROCESSED_PATH, rate=22050)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=16)
    _, data_inputs, data_outputs = next(iter(dataloader))
    data_outputs = data_outputs.detach().numpy()
    spectro = torchaudio.transforms.MelSpectrogram(n_fft=1024, hop_length=256,
                                                   sample_rate=22050,
                                                   power=2, n_mels=64)

    data_inputs = spectro(data_inputs).detach().numpy()
    data_inputs = data_inputs.flatten()
    num_generations = 50
    num_parents_mating = 4

    fitness_function = fitness_func

    sol_per_pop = 8
    num_genes = len(data_inputs) * 8
    print("Num genes: ", num_genes)

    init_range_low = -2
    init_range_high = 5

    parent_selection_type = "sss"
    keep_parents = 1

    crossover_type = "single_point"

    mutation_type = "random"
    mutation_percent_genes = 10

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes,
                           on_generation=callback_generation)

    ga_instance.run()
    ga_instance.plot_fitness()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))