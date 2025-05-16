import pathlib
import os

import numpy as np
import basegraph

from pybigue.models import angular_distance_1d
from results.paths import ResultPaths, write_dataclass, write_graph
from pybigue.utils import align_theta, angle_modulo, sample_truncated_normal, sample_uniform, sample_truncated_pareto
from pybigue.embedding_info import EmbeddingParameters, GraphInformation


def generate_conflicting(embedding1, embedding2):
    average_degree = (np.average(embedding1.kappa)+np.average(embedding2.kappa))/2
    n = len(embedding1.theta)
    R_div_mu1 = n * average_degree / (embedding1.beta * np.sin(np.pi / embedding1.beta))
    R_div_mu2 = n * average_degree / (embedding2.beta * np.sin(np.pi / embedding2.beta))

    graph = basegraph.core.UndirectedGraph(n)
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.rand() <= 0.5:
                chi = R_div_mu1 * angular_distance_1d(
                    embedding1.theta[i], embedding1.theta[j]) / embedding1.kappa[i] / embedding1.kappa[j]
                prob = 1 / (1 + chi**embedding1.beta)
            else:
                chi = R_div_mu2 * angular_distance_1d(
                    embedding2.theta[i], embedding2.theta[j]) / embedding2.kappa[i] / embedding2.kappa[j]
                prob = 1 / (1 + chi**embedding2.beta)

            if np.random.rand() <= prob:
                graph.add_edge(i, j)
    return graph


if __name__ == "__main__":
    config_name = "conflicting"
    n = 30
    kappa_min = 4
    kappa_max = 10
    gamma = 3

    output_file = str(pathlib.Path(__file__).parent.joinpath(".data", "conflicting.txt").resolve())

    # Graph generation

    np.random.seed(69)
    embedding1 = EmbeddingParameters(
        theta=sample_uniform(-np.pi, np.pi, n),
        kappa=sample_truncated_pareto(kappa_min, kappa_max, gamma, size=n).tolist(),
        beta=sample_truncated_normal(3, 1, size=1, lower_bound=1))
    theta2 = np.copy(embedding1.theta)
    theta2[0] = angle_modulo(theta2[0] + 2)
    embedding2 = EmbeddingParameters(
        theta=theta2,
        kappa=embedding1.kappa,
        beta=embedding1.beta)

    graph = generate_conflicting(embedding1, embedding2)
    graph_info = GraphInformation.from_degrees(graph.get_degrees())
    embedding1.theta = align_theta(embedding1.theta, *graph_info.fixed_vertices)
    embedding2.theta = align_theta(embedding2.theta, *graph_info.fixed_vertices)

    print(len(basegraph.metrics.find_weakly_connected_components(graph)), "connected component")
    write_graph(graph, output_file)

    path_manager = ResultPaths(config_name)


    if not os.path.isdir(path_manager.config_output_path):
        os.makedirs(path_manager.config_output_path)
    write_dataclass(graph_info, path_manager.graph_information_path)
    write_dataclass(embedding1,
                    path_manager.groundtruth_embedding_path+"_1")
    write_dataclass(embedding2,
                    path_manager.groundtruth_embedding_path+"_2")
