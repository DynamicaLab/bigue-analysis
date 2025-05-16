import os
import re
import subprocess
import tempfile
import warnings
import pathlib
import itertools

import basegraph
import numpy as np
from sklearn.metrics import roc_auc_score

from pybigue.embedding_info import EmbeddingsContainer, GraphInformation, EmbeddingParameters, replace_known_parameters
from pybigue.sampling import read_sample, write_sample
from pybigue.models import S1Model, angular_distance


def merge_chains(chain_samples):
    if len(chain_samples) == 0:
        warnings.warn("No chain found to merge.")
        return EmbeddingsContainer()
    return EmbeddingsContainer(
            thetas=list(itertools.chain.from_iterable([x for x in map(lambda x: x.thetas, chain_samples.values()) if x is not None])),
            kappas=list(itertools.chain.from_iterable([x for x in map(lambda x: x.kappas, chain_samples.values()) if x is not None])),
            betas=list(itertools.chain.from_iterable([x for x in map(lambda x: x.betas, chain_samples.values()) if x is not None])),
        )


def find_automorphisms(graph_filename, n):
    project_dir = str(pathlib.Path(__file__).parent.parent.resolve())
    nauty_program = os.path.join(project_dir, "nauty-automorph", "nauty-automorph")

    if os.name == 'nt':
        nauty_program += ".exe"

    output_file = tempfile.mkstemp()[1]
    subprocess.run([nauty_program, str(n), graph_filename, output_file])
    res = np.loadtxt(output_file).astype(np.int32)
    os.remove(output_file)
    if res.ndim == 1:
        res = [res]
    return list(map(lambda p: np.argsort(p), res))


adjusted_sample_format = "adjusted_{}_{}.npy"
adjusted_sample_regex = re.compile(r"adjusted_.+_theta.npy$")

def read_adjusted_sample(path_manager, reference_theta):
    chain_samples = read_sample(path_manager.cache_dir_path, file_format=adjusted_sample_format)

    if chain_samples == {}:
        if not os.path.isdir(path_manager.cache_dir_path):
            os.makedirs(path_manager.cache_dir_path)
        print(f'Caching aligned samples of "{path_manager.sample_prefix}" init')

        chain_samples = read_sample(path_manager.sample_dir_path)
        automorphisms = find_automorphisms(path_manager.graph_path, len(reference_theta))
        for chain, samples in chain_samples.items():
            for i, sample in enumerate(samples):
                ideal_symmetry = S1Model.find_ideal_symmetry(sample.theta, reference_theta, automorphisms)
                chain_samples[chain][i] = S1Model.apply_symmetry(sample, ideal_symmetry)
            write_sample(samples, chain, path_manager.cache_dir_path, adjusted_sample_format)

    return chain_samples


loglikelihood_file_format = "loglikelihoods_{}.npy"

def get_matrix_prob(parameters: EmbeddingParameters, average_degree):
    if parameters.theta is None or parameters.kappa is None or parameters.beta is None:
        raise ValueError

    n = len(parameters.theta)
    probs = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            prob = S1Model.get_edge_prob(parameters,  average_degree, i, j)
            probs[i][j] = prob
            probs[j][i] = prob
    return probs


def get_auc(graph, edge_probs: np.ndarray):
    true_classes = graph.get_adjacency_matrix(False)
    return roc_auc_score(np.ravel(true_classes), np.ravel(edge_probs))


def get_chain_loglikelihoods(graph, samples, chain, known_parameters, path_manager):
    graph_info = GraphInformation.from_degrees(graph.get_degrees())

    loglikelihood_file = os.path.join(path_manager.cache_dir_path, loglikelihood_file_format.format(chain))
    if os.path.isfile(loglikelihood_file):
        loglikelihoods = np.load(loglikelihood_file)
    else:
        if not os.path.isdir(path_manager.cache_dir_path):
            os.makedirs(path_manager.cache_dir_path)
        print(f'Caching loglikelihoods of chain {chain} of "{path_manager.sample_prefix}" init')
        inferred_parameters = [parameter for parameter, sample in samples.items() if sample is not None]
        if inferred_parameters == []:
            raise ValueError("No parameter was inferred.")

        adjacency = np.array(graph.get_adjacency_matrix(False))
        loglikelihoods = np.array([
            S1Model.loglikelihood(adjacency, graph_info.average_degree, **replace_known_parameters(sample, known_parameters).as_dict())
            for sample in samples
        ])
        np.save(loglikelihood_file, loglikelihoods)
    return loglikelihoods


def generate_graph(embedding_params: EmbeddingParameters,
                   average_degree: float):
    """Sample graph from S^1 model."""
    beta = embedding_params.beta
    theta = embedding_params.theta
    kappa = embedding_params.kappa

    if theta is None:
        raise ValueError("Theta cannot be None in graph generation.")
    if kappa is None:
        raise ValueError("Kappa cannot be None in graph generation.")
    if beta is None:
        raise ValueError("Beta cannot be None in graph generation.")

    n = len(theta)

    R_div_mu = n * average_degree / (beta * np.sin(np.pi / beta))

    graph = basegraph.core.UndirectedGraph(n)
    for i in range(n):
        for j in range(i + 1, n):
            chi = R_div_mu * angular_distance(
                theta[i], theta[j]) / kappa[i] / kappa[j]

            if np.random.rand() <= 1 / (1 + chi**beta):
                graph.add_edge(i, j)
    return graph
