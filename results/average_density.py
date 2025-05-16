import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from matplotlib import pyplot

from pybigue.models import S1Model
from paths import ResultPaths, read_dataclass, read_graph
from pybigue.embedding_info import EmbeddingParameters, GraphInformation
from plot_tools import darken_color, figure_dir, all_colors, hist_colors
from modelingtools import merge_chains, read_adjusted_sample


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
    percentiles = [25, 50, 75]

    graph = read_graph(path_manager.graph_path)
    graph_info = read_dataclass(GraphInformation, path_manager.graph_information_path)
    has_groundtruth = os.path.isfile(path_manager.groundtruth_embedding_path)
    if has_groundtruth:
        reference = read_dataclass(EmbeddingParameters, path_manager.groundtruth_embedding_path)
    else:
        print("Using Mercator as reference.")
        reference = read_dataclass(EmbeddingParameters, path_manager.mercator_embedding_path)
    if reference is None or reference.theta is None:
        raise ValueError("Couldn't find a reference embedding.")


    aligned_sample = merge_chains(read_adjusted_sample(path_manager, reference.theta))

    fig, ax = pyplot.subplots(1, figsize=(13, 13), subplot_kw={"projection": "polar"})

    mus = S1Model.get_mu(np.array(aligned_sample.betas), graph_info.average_degree)
    kappa_min_h2 = 1
    h2_rad = lambda t: S1Model.get_H2_radii(t, kappa_min_h2, mus, n=graph_info.n)

    marker_size = 350
    colors = [darken_color(all_colors[0], light_correction=1.5, saturation_correction=1.1), all_colors[2], all_colors[3]]
    pale_colors = [darken_color(color, light_correction=1.3, saturation_correction=1.1)
                    if i>0 else hist_colors["posterior"] for i, color in enumerate(colors)]
    radii_med = []
    thetas_med = []
    for v, (theta_i, kappa_i, theta_0) in enumerate(zip(
                np.array(aligned_sample.thetas).T, np.array(aligned_sample.kappas).T, reference.theta)):
        ax.scatter(theta_i, h2_rad(kappa_i),
                   color="tab:blue", marker=".", s=marker_size, alpha=0.05, zorder=1, edgecolor="none")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, None)

    output_dir = figure_dir()
    # fig.savefig(str(output_dir.joinpath(f"hyperbolic_error_bars_{config_name}.png")),
                # transparent=True,
                # dpi=400,
                # bbox_inches="tight")
    fig.tight_layout()
    pyplot.show()
