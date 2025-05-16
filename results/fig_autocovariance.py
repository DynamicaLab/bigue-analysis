import os

import numpy as np
from matplotlib import pyplot

from paths import ResultPaths, read_dataclass, read_graph
from plot_tools import figure_dir, column_width, alg_names, alg_colors
from pybigue.embedding_info import EmbeddingParameters, GraphInformation
from pybigue.metrics import normalized_reg_autocovariance, normalized_circ_autocovariance
from pybigue.sampling import read_sample


init_full_names = {
    "random": "Random initialization",
    "mercator": "Mercator initialization",
    "groundtruth": "Ground truth initialization"
}

if __name__ == "__main__":
    config_name = "30v" # can be any other config based on that graph
    path_manager = ResultPaths(config_name)

    graph = read_graph(path_manager.graph_path)
    graph_info = read_dataclass(GraphInformation, path_manager.graph_information_path)
    mercator_parameters = read_dataclass(EmbeddingParameters, path_manager.mercator_embedding_path)

    fig, ax = pyplot.subplots(1, figsize=(column_width, 0.42*column_width), sharey=True, sharex=True)

    reference_theta = np.array(mercator_parameters.theta)
    sorted_index = np.argsort(reference_theta)

    step = 10
    lags = np.arange(1, 500+1, step)
    warmup = 6000

    colors = [alg_colors["posterior"], alg_colors["rw"], alg_colors["hmc"]]
    for config_name, color in zip(["30v", "30v_rw", "30v_hmc"], colors):
        path_manager = ResultPaths(config_name)

        path_manager.sample_prefix = "groundtruth"
        if not os.path.isdir(path_manager.sample_dir_path):
            continue
        chain_samples = read_sample(path_manager.sample_dir_path)
        if chain_samples == {}:
            continue

        yvals = np.average([
                [[normalized_circ_autocovariance(np.array(sample.thetas)[warmup:, v], lag) for lag in lags] for v in graph if v!=graph_info.fixed_vertices[0]]
                + [[normalized_reg_autocovariance(np.array(sample.kappas)[warmup:, v], lag) for lag in lags] for v in graph]
                + [[normalized_reg_autocovariance(np.array(sample.betas)[warmup:], lag) for lag in lags]]
                for sample in chain_samples.values()],
               axis=(0, 1)
           )
        ax.plot(10*lags, yvals, label=alg_names[config_name], color=color, alpha=0.7)
    ax.set_xlim((0, None))
    ax.set_ylim((0, 1))
    ax.set_xlabel("Lags")
    ax.set_ylabel("Normalized\nautocovariance")

    output_dir = figure_dir()
    fig.savefig(str(output_dir.joinpath("autocovariance_30v.pdf")),
                bbox_inches="tight")

    fig.tight_layout()
    pyplot.show()
