import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from matplotlib import pyplot

from paths import ResultPaths, read_dataclass, read_graph
from pybigue.models import S1Model
from pybigue.sampling import read_sample
from pybigue.embedding_info import EmbeddingParameters, GraphInformation

from modelingtools import get_chain_loglikelihoods, get_chain_loglikelihoods

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config_file", help="Config file path")
    parser.add_argument("-i", "--init", nargs="+", default=["random", "groundtruth", "mercator"],
                        help="Initialization", choices=["random", "groundtruth", "mercator"])
    args = parser.parse_args()
    config_name = os.path.splitext(Path(args.config_file).name)[0]

    path_manager = ResultPaths(config_name)

    graph = read_graph(path_manager.graph_path)
    graph_info = read_dataclass(GraphInformation, path_manager.graph_information_path)

    fig, ax = pyplot.subplots(1)

    if os.path.isfile(path_manager.groundtruth_embedding_path):
        groundtruth_parameters = read_dataclass(EmbeddingParameters, path_manager.groundtruth_embedding_path)
        ground_truth_likelihood = S1Model.loglikelihood(np.array(graph.get_adjacency_matrix(False)), graph_info.average_degree,
                                                        **groundtruth_parameters.as_dict())
        ax.axhline(ground_truth_likelihood, label="Ground truth", color="k", lw=2.5)
    else:
        groundtruth_parameters = None

    colors = ["#EEAAD7", "#9567E0", "#89B8D2", "#DFA953"]
    for color, init in zip(colors, args.init):
        path_manager.sample_prefix = init
        if not os.path.isdir(path_manager.sample_dir_path):
            continue

        chain_samples = read_sample(path_manager.sample_dir_path)

        for i, (chain, samples) in enumerate(chain_samples.items()):
            loglikelihoods = get_chain_loglikelihoods(graph, samples, chain, EmbeddingParameters(), path_manager)
            ax.plot(np.arange(len(loglikelihoods)), loglikelihoods, label=f"{init} init" if i==0 else None, color=color)

        ax.set_ylabel("Loglikelihood")
        ax.set_xlabel("Iterations")
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    mercator_parameters = read_dataclass(EmbeddingParameters, path_manager.mercator_embedding_path)
    ax.axhline(S1Model.loglikelihood(np.array(graph.get_adjacency_matrix(False)), graph_info.average_degree, **mercator_parameters.as_dict()),
               label="Mercator", alpha=0.5, color="gray", ls="--")

    ax.legend()
    fig.tight_layout()

    pyplot.show()
