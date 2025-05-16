import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from matplotlib import pyplot

from modelingtools import merge_chains, read_adjusted_sample, find_automorphisms
from paths import ResultPaths, read_dataclass, read_graph
from plot_tools import darken_color, figure_dir, all_colors, hist_colors
from pybigue.embedding_info import EmbeddingParameters, GraphInformation
from pybigue.models import S1Model
from pybigue.utils import angle_modulo


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

    shown_vertices = [1, 5, 13]
    marker_size = 350
    colors = [darken_color(all_colors[0], light_correction=1.5, saturation_correction=1.1), all_colors[2], all_colors[3]]
    pale_colors = [darken_color(color, light_correction=1.3, saturation_correction=1.1)
                    if i>0 else hist_colors["posterior"] for i, color in enumerate(colors)]
    radii_med = []
    thetas_med = []
    for v, (theta_i, kappa_i, theta_0) in enumerate(zip(
                np.array(aligned_sample.thetas).T, np.array(aligned_sample.kappas).T, reference.theta)):

        theta_low, theta_median, theta_high = np.percentile(center_at_reference(theta_i, theta_0), percentiles)
        r_low, r_median, r_high = np.percentile(h2_rad(kappa_i), percentiles)

        radii_med.append(r_median)
        thetas_med.append(angle_modulo(theta_median))

        if v in shown_vertices:
            ind = shown_vertices.index(v)
            ax.scatter(theta_i, h2_rad(kappa_i),
                       color=pale_colors[ind], marker=".", s=marker_size, alpha=0.2, zorder=1, edgecolor="none")
            ax.scatter(theta_median, r_median,
                       marker="o", color=colors[ind], s=marker_size, alpha=1, zorder=3, edgecolor=midblack, linewidth=2)
        else:
            ax.scatter(theta_median, r_median,  marker="o", color=midblack,
                       s=marker_size, alpha=1, zorder=2, edgecolor="none")
            # bar_style = {"ls": "-", "color": "k", "marker": "None", "zorder": 3}
            # between_theta = np.arange(theta_low, theta_high+0.05, 0.05)
            # ax.plot(center_at_reference(between_theta, theta_0), np.full_like(between_theta, r_median), **bar_style)
            # ax.plot([theta_median]*2, [r_low, r_high], **bar_style)

    for u in graph:
        for v in graph.get_neighbours(u):
            if u<v:
                xs, ys, zs = get_geodesic_coordinates(radii_med[u], thetas_med[u], radii_med[v], thetas_med[v], 50)
                radii = np.arccosh(zs)
                thetas = np.arctan2(ys, xs)
                ax.plot(thetas, radii, color=midblack, zorder=1, alpha=0.5, lw=3)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, None)


    mercator_embedding = read_dataclass(EmbeddingParameters, path_manager.mercator_embedding_path)
    if has_groundtruth:
        mercator_embedding = read_dataclass(EmbeddingParameters, path_manager.mercator_embedding_path)
        automorphisms = find_automorphisms(path_manager.graph_path, graph_info.n)
        mercator_embedding.theta = np.asarray(mercator_embedding.theta)
        mercator_embedding.kappa = np.asarray(mercator_embedding.kappa)
        ideal_symmetry = S1Model.find_ideal_symmetry(mercator_embedding.theta, reference.theta, automorphisms)
        remapped_mercator = S1Model.apply_symmetry(mercator_embedding, ideal_symmetry)

        # ax.scatter(
            # remapped_mercator.theta,
            # S1Model.get_H2_radii(
                # np.array(remapped_mercator.kappa),
                # kappa_min_h2,
                # S1Model.get_mu(remapped_mercator.beta, graph_info.average_degree),
                # n=graph_info.n
            # ),
            # s=marker_size,
            # color="red",
            # zorder=4,
            # marker="*"
        # )

    output_dir = figure_dir()
    fig.savefig(str(output_dir.joinpath(f"hyperbolic_error_bars_{config_name}.png")),
                transparent=True,
                dpi=400,
                bbox_inches="tight")
    fig.tight_layout()
    pyplot.show()
