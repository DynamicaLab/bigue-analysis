import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from matplotlib import pyplot

from pybigue.models import S1Model
from paths import ResultPaths, read_dataclass, read_graph
from pybigue.embedding_info import EmbeddingParameters, GraphInformation
from plot_tools import figure_dir


def center_at_reference(theta, reference_theta):
    adjusted_theta = np.copy(theta)
    adjusted_theta[adjusted_theta > (reference_theta+np.pi)] -= 2*np.pi
    adjusted_theta[adjusted_theta < (reference_theta-np.pi)] += 2*np.pi
    return adjusted_theta


def get_geodesic_coordinates(r_vertex1, t_vertex1, r_vertex2, t_vertex2, nb_points=10):
    s = np.transpose(np.tile(np.linspace(0, 1, nb_points), (3, 1)))

    x = np.sinh(r_vertex1) * np.cos(t_vertex1)
    y = np.sinh(r_vertex1) * np.sin(t_vertex1)
    z = np.cosh(r_vertex1)
    p1 = np.tile(np.array([x/z, y/z, 1]), (nb_points, 1))

    x = np.sinh(r_vertex2) * np.cos(t_vertex2)
    y = np.sinh(r_vertex2) * np.sin(t_vertex2)
    z = np.cosh(r_vertex2)
    p2 = np.tile(np.array([x/z, y/z, 1]), (nb_points, 1))

    geodesic = s * p1 + (1 - s) * p2
    a = np.sqrt(np.ones(nb_points) - geodesic[:, 0]**2 - geodesic[:, 1]**2)

    return geodesic.T / a


if __name__ == "__main__":
    midblack = "#484848"

    parser = ArgumentParser()
    parser.add_argument("config_file", help="Config file path")
    parser.add_argument("-i", "--init", default="random", help="Initialization", choices=["random", "groundtruth", "mercator"])
    args = parser.parse_args()
    config_name = os.path.splitext(Path(args.config_file).name)[0]

    path_manager = ResultPaths(config_name)
    path_manager.sample_prefix = args.init

    graph = read_graph(path_manager.graph_path)
    graph_info = read_dataclass(GraphInformation, path_manager.graph_information_path)
    if not os.path.isfile(path_manager.groundtruth_embedding_path):
        print("No ground truth embedding found")
        exit()
    reference = read_dataclass(EmbeddingParameters, path_manager.groundtruth_embedding_path)

    fig, ax = pyplot.subplots(1, figsize=(13, 13), subplot_kw={"projection": "polar"})
    fig2, ax2 = pyplot.subplots(1, figsize=(13, 13), subplot_kw={"projection": "polar"})

    mus = S1Model.get_mu(reference.beta, graph_info.average_degree)
    vertex_radii = S1Model.get_H2_radii(reference.kappa, 1, mus, n=graph_info.n)

    marker_size = 1300
    for u in graph:
        for v in graph.get_neighbours(u):
            if u<v:
                xs, ys, zs = get_geodesic_coordinates(vertex_radii[u], reference.theta[u], vertex_radii[v], reference.theta[v], 50)
                radii = np.arccosh(zs)
                thetas = np.arctan2(ys, xs)
                ax.plot(thetas, radii, color=midblack, zorder=1, alpha=0.5, lw=3)
    ax.scatter(reference.theta, vertex_radii,
               color=midblack, marker=".", s=marker_size, zorder=1, edgecolor="none")
    ax2.scatter(reference.theta, vertex_radii,
               color=midblack, marker=".", s=marker_size, zorder=1, edgecolor="none")

    for a in [ax, ax2]:
        a.set_xticks([])
        a.set_yticks([])
        a.set_ylim(0, None)

    output_dir = figure_dir()
    fig.savefig(str(output_dir.joinpath(f"hyperbolic_{config_name}_groundtruth.png")),
                transparent=True,
                dpi=400,
                bbox_inches="tight")
    fig2.savefig(str(output_dir.joinpath(f"hyperbolic_{config_name}_groundtruth_positions.png")),
                 transparent=True,
                 dpi=400,
                 bbox_inches="tight")
    fig.tight_layout()
    fig2.tight_layout()
    pyplot.show()
