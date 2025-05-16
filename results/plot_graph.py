from argparse import ArgumentParser
import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import networkx as nx

from paths import ResultPaths, read_graph
from plot_tools import figure_dir


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config_file", help="Config file path")
    args = parser.parse_args()
    config_name = os.path.splitext(Path(args.config_file).name)[0]

    path_manager = ResultPaths(config_name)
    graph = read_graph(path_manager.graph_path)
    nx_graph = nx.from_numpy_array(np.array(graph.get_adjacency_matrix(False)))
    pos = nx.kamada_kawai_layout(nx_graph)
    nx.draw_networkx_edges(nx_graph, pos)
    nx.draw_networkx_nodes(nx_graph, pos, node_size=200, node_color="k", edgecolors="k",)
    # nx.draw_networkx_labels(nx_graph, pos, font_size=14)
    plt.axis("off")
    plt.tight_layout()

    fig_path = figure_dir()
    plt.savefig(str(fig_path.joinpath(f"{config_name}_graph.pdf")),
                bbox_inches="tight")
    plt.show()
