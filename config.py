import os
from pathlib import Path
from dataclasses import dataclass

import yaml
import numpy as np

from pybigue.embedding_info import Hyperparameters
from results.paths import read_graph


@dataclass
class AlgorithmParameters:
    sample_size: int
    kernels: dict
    warmup: int = 200
    thin: int = 1
    chain_number: int = 4
    filter_graph: bool = False

    def validate(self):
        if self.sample_size < 1:
            raise ValueError("Sample size must be at least 1.")
        if self.chain_number < 1:
            raise ValueError("The number of chains must be at least 1.")
        if self.warmup < 0:
            raise ValueError("Warmup must not be negative.")

        sum_of_probs = 0
        for kernel_name, params in self.kernels.items():
            if (prob := params.get("prob")) is None:
                raise ValueError(f"Kernel \"{kernel_name}\"'s \"prob\" is missing.")
            sum_of_probs += prob
        if abs(sum_of_probs - 1) > 1e-8:
            raise ValueError("Sum of kernel probabilities is not 1.")

        if self.thin <= 0:
            raise ValueError("Thinning must be at least 1.")
        return self


@dataclass
class GenerationParameters:
    n: int
    kappa_min: float
    kappa_max: float
    beta: float
    gamma: float

    def validate(self):
        if self.n < 2:
            raise ValueError("There must be at least two vertices.")
        if self.kappa_min > self.kappa_max:
            raise ValueError(
                "Maximum kappa must be greater that minimum kappa.")

        for value, name in zip(
            [self.kappa_min, self.kappa_max, self.beta, self.gamma],
            ["Minimum kappa", "Maximum kappa", "beta", "gamma"]):

            if value < 0:
                raise ValueError(f"{name} must be greater than 0.")
        return self

def hyperparameters_with_defaults(vertex_number, gamma=4, radius=None, beta_average=3, beta_std=2):
    if radius is None:
        radius = vertex_number/2*np.pi
    return Hyperparameters(gamma=gamma, radius=radius, beta_average=beta_average, beta_std=beta_std)


def parse_config(filename):
    with open(filename, "r") as file_stream:
        config = yaml.safe_load(file_stream)

    is_synthetic_config = config.get("generation") is not None
    is_data_config = config.get("graph") is not None
    if is_synthetic_config and is_data_config:
        raise ValueError("Config cannot be applied both on synthetic and empirical graphs.")
    if not is_synthetic_config and not is_data_config:
        raise ValueError("Config has to be applied on either synthetic or empirical graphs.")

    if is_data_config:
        return parse_data_config(filename)
    return parse_synthetic_config(filename)

def parse_synthetic_config(filename):
    with open(filename, "r") as file_stream:
        config = yaml.safe_load(file_stream)
        file_path = Path(filename)
        filename_no_extension = os.path.splitext(file_path.name)[0]

        hyperparameters = config.get("hyperparameters")
        return {
            "name": filename_no_extension,
            "seed": config.get("seed"),
            "algorithm": AlgorithmParameters(**config["algorithm"]),
            "generation": (gen_params := GenerationParameters(**config["generation"])),
            "hyperparameters": hyperparameters_with_defaults(gen_params.n) if hyperparameters is None else
                                hyperparameters_with_defaults(gen_params.n, **hyperparameters),
        }

def parse_data_config(filename):
    with open(filename, "r") as file_stream:
        config = yaml.safe_load(file_stream)
        file_path = Path(filename)
        filename_no_extension = os.path.splitext(file_path.name)[0]

        graph = read_graph(config["graph"])
        hyperparameters = config.get("hyperparameters")
        return {
            "name": filename_no_extension,
            "seed": config.get("seed"),
            "graph": graph,
            "algorithm": AlgorithmParameters(**config["algorithm"]),
            "hyperparameters": hyperparameters_with_defaults(graph.get_size()) if hyperparameters is None else
                                hyperparameters_with_defaults(graph.get_size(), **hyperparameters),
        }
