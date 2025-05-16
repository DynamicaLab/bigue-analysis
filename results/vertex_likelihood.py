import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colormaps

from paths import ResultPaths, read_dataclass, read_graph
from plot_tools import format_ticks, figure_dir, column_width, darken_color
from pybigue.embedding_info import EmbeddingParameters, GraphInformation
from pybigue.models import S1Model


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config_file", help="Config file path")
    args = parser.parse_args()

    config_name = os.path.splitext(Path(args.config_file).name)[0]
    path_manager = ResultPaths(config_name)

    graph = read_graph(path_manager.graph_path)
    graph_info = read_dataclass(GraphInformation, path_manager.graph_information_path)
    if os.path.isfile(path_manager.groundtruth_embedding_path):
        groundtruth_parameters = read_dataclass(EmbeddingParameters, path_manager.groundtruth_embedding_path)
    else:
        raise ValueError("Cannot generate figure without known ground truth.")


    fig, ax = plt.subplots(1, figsize=(column_width, 0.65*column_width))

    xvalues = np.linspace(-np.pi, np.pi, 400)
    degrees = graph.get_degrees()

    median = np.median(degrees)
    v = None
    for i, k in enumerate(degrees):
        if k == median:
            v = i
            break

    cmap = colormaps["Blues"]
    colors = list(map(cmap, np.linspace(0.2, 1, 3)))[::-1]
    for vertex, pos, color in zip([np.argmax(degrees), v, np.argmin(degrees)],
                                  ["highest", "median", "lowest"],
                                  colors):
        gt = EmbeddingParameters(np.copy(groundtruth_parameters.theta), np.copy(groundtruth_parameters.kappa), groundtruth_parameters.beta)
        non_neighbours = [v for v in range(graph.get_size()) if v not in graph.get_neighbours(vertex)]
        adjusted_xvalues = np.sort(np.concatenate(
            [gt.theta[v] + np.linspace(-1e-3, 1e-3, 101)
             for v in non_neighbours]
            + [xvalues]
        ))
        yvalues = []
        for new_value in adjusted_xvalues:
            gt.theta[vertex] = new_value
            yvalues.append(S1Model.loglikelihood(np.array(graph.get_adjacency_matrix(False)), graph_info.average_degree, **gt.as_dict()))
        ax.plot(adjusted_xvalues, yvalues, label=f"Degree {degrees[vertex]} ({pos})", lw=1,
                color=darken_color(color, light_correction=0.9))

    ax.set_ylabel("Log-likelihood")
    ax.set_xlabel("Vertex position")
    ax.set_xticks(*format_ticks(4))
    ax.set_xlim((-np.pi-.1, np.pi+.1))
    ax.legend(loc="lower center")
    fig.tight_layout()

    output_dir = figure_dir()
    fig.savefig(str(output_dir.joinpath("vertex_likelihood.pdf")))
    plt.show()
