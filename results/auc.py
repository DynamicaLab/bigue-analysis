import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from paths import ResultPaths, read_dataclass, read_graph
from pybigue.sampling import read_sample
from pybigue.embedding_info import EmbeddingParameters, GraphInformation

from modelingtools import get_auc, get_matrix_prob, merge_chains
from plot_tools import column_width, figure_dir, hist_colors, alg_colors

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config_file", help="Config file path")
    parser.add_argument("-i", "--init", default="random",
                        help="Initialization", choices=["random", "groundtruth", "mercator"])
    args = parser.parse_args()
    config_name = os.path.splitext(Path(args.config_file).name)[0]

    path_manager = ResultPaths(config_name)

    graph = read_graph(path_manager.graph_path)
    graph_info = read_dataclass(GraphInformation, path_manager.graph_information_path)

    fig, ax = plt.subplots(1, figsize=(.5*column_width, column_width/2.5), tight_layout=True)
    if os.path.isfile(path_manager.groundtruth_embedding_path):
        groundtruth_parameters = read_dataclass(EmbeddingParameters, path_manager.groundtruth_embedding_path)
        gt_auc = get_auc(graph, get_matrix_prob(groundtruth_parameters, graph_info.average_degree))
        ax.axvline(gt_auc, color="k", lw=2.5, ls="--")
    else:
        groundtruth_parameters = None

    path_manager.sample_prefix = args.init
    samples = merge_chains(read_sample(path_manager.sample_dir_path))

    mercator_parameters = read_dataclass(EmbeddingParameters, path_manager.mercator_embedding_path)
    mercator_auc = get_auc(graph, get_matrix_prob(mercator_parameters, graph_info.average_degree))
    bigue_aucs = [get_auc(graph, get_matrix_prob(embedding, graph_info.average_degree)) for embedding in samples]

    ax.hist(bigue_aucs, bins=20, color=hist_colors["posterior"], alpha=0.5)
    ax.axvline(np.median(bigue_aucs), color=alg_colors["posterior"], lw=2.5, ls="--")
    ax.axvline(mercator_auc, color=alg_colors["mercator"], lw=2.5, ls="--")
    ax.spines[["left"]].set_visible(False)
    ax.set_yticks([])
    ax.set_xlabel("AUC")

    fig.tight_layout()
    output_dir = figure_dir()
    fig.savefig(str(output_dir.joinpath(f"auc_{config_name}.pdf")))

    plt.show()
