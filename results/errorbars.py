import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from matplotlib import pyplot
from matplotlib.patches import Patch

from modelingtools import merge_chains, read_adjusted_sample, find_automorphisms
from paths import ResultPaths, read_dataclass, read_graph
from plot_tools import format_ticks, alg_colors, hist_colors, figure_dir, column_width, dark_reference
from pybigue.embedding_info import EmbeddingParameters, GraphInformation
from pybigue.models import S1Model
from pybigue.sampling import read_sample


def center_at_reference(theta, reference_theta):
    adjusted_theta = np.copy(theta)
    adjusted_theta[adjusted_theta > (reference_theta+np.pi)] -= 2*np.pi
    adjusted_theta[adjusted_theta < (reference_theta-np.pi)] += 2*np.pi
    return adjusted_theta


def plot_kappas(ax, estimation, ground_truth, sample_size, color, use_bars=True, percentiles=[25, 50, 75], **kwargs):
    if sample_size > 1 and not use_bars:
        ax.scatter(
            np.full_like(estimation, ground_truth),
            estimation,
            alpha=1 / sample_size,
            color=color,
            marker=".",
            **kwargs,
        )
    else:
        if sample_size > 1:
            low, median, high = np.percentile(estimation, percentiles)
            bar_style = kwargs | {"ls": "-", "color": color, "marker": "None"}
            bar_style.pop("s")
            ax.plot([ground_truth]*2, [median, low], **bar_style)
            ax.plot([ground_truth]*2, [median, high], **bar_style)
        else:
            median = estimation
        ax.scatter(ground_truth, median, marker=".", color=color, **kwargs)


if __name__ == "__main__":
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
        groundtruth = read_dataclass(EmbeddingParameters, path_manager.groundtruth_embedding_path)
    else:
        print("Using Mercator as reference.")
        groundtruth = read_dataclass(EmbeddingParameters, path_manager.mercator_embedding_path)

    sorted_index = np.argsort(groundtruth.theta)

    sample = merge_chains(read_sample(path_manager.sample_dir_path))
    aligned_sample = merge_chains(read_adjusted_sample(path_manager, groundtruth.theta))
    mercator_embedding = read_dataclass(EmbeddingParameters, path_manager.mercator_embedding_path)

    automorphisms = find_automorphisms(path_manager.graph_path, graph_info.n)
    mercator_embedding.theta = np.asarray(mercator_embedding.theta)
    mercator_embedding.kappa = np.asarray(mercator_embedding.kappa)
    ideal_symmetry = S1Model.find_ideal_symmetry(mercator_embedding.theta, groundtruth.theta, automorphisms)
    remapped_mercator = S1Model.apply_symmetry(mercator_embedding, ideal_symmetry)


    # fig, axes = pyplot.subplots(3, tight_layout=True, figsize=(0.55*column_width, column_width/2.5*3))
    fig, axes = pyplot.subplots(3, tight_layout=True, figsize=(0.67*column_width, 1.5*column_width))
    markersize = 7
    bar_linewidth = 0.7

    mercator_plot_args = {"color": alg_colors["mercator"], "marker": "x", "zorder": 1, "linewidth":.7}

    # Theta bars
    ax = axes[0]
    for theta_i, theta_0 in zip(np.array(aligned_sample.thetas).T, groundtruth.theta):
        adjusted_theta_i = center_at_reference(theta_i, theta_0)
        low, median, high = np.percentile(adjusted_theta_i, percentiles)
        posterior_handle = ax.scatter(theta_0, median, marker=".", color=alg_colors["posterior"],
                                      s=markersize, alpha=1, zorder=2)
        bar_style = {"ls": "-", "color": alg_colors["posterior"], "marker": "None", "zorder": 3, "linewidth": bar_linewidth}

        ax.plot([theta_0]*2, [median, low], **bar_style)
        ax.plot([theta_0]*2, [median, high], **bar_style)

    angles = np.linspace(-np.pi, np.pi, 2)
    ax.plot(angles, angles, color="gray", alpha=0.3, ls='-', zorder=0)

    if has_groundtruth:
        ax.scatter(groundtruth.theta[sorted_index],
                center_at_reference(remapped_mercator.theta[sorted_index], groundtruth.theta[sorted_index]),
                s=1.5*markersize,
                **mercator_plot_args)

    lims = (-np.pi-0.1, np.pi+0.1)
    ax.set_xlim(lims)
    ax.axhline(-np.pi, color="gray", alpha=0.1, ls="--")
    ax.axhline(np.pi, color="gray", alpha=0.1, ls="--")
    if has_groundtruth:
        ax.set_xlabel("Ground truth $\\theta$")
    else:
        ax.set_xlabel("Mercator $\\theta$")

    ax.set_xticks(*format_ticks(2))
    ax.set_yticks(*format_ticks(2))
    ax.set_ylabel("Sample $\\theta$")

    # Kappa bars
    ax = axes[1]

    reference = graph.get_degrees()
    kappa = np.array(sample.kappas)
    sample_size, n = kappa.shape
    for kappa_i, original_kappa in zip(kappa.T, reference):
        plot_kappas(ax, kappa_i, original_kappa, sample_size, alg_colors["posterior"],
                    use_bars=True, percentiles=percentiles, zorder=4, s=markersize, linewidth=bar_linewidth)

    ax.scatter(reference, mercator_embedding.kappa, s=1.5*markersize, **mercator_plot_args)

    lims = ax.get_xlim()
    ax.plot(lims, lims, color="gray", ls='-', alpha=0.3, zorder=0)
    ax.set_xlabel("Degrees")
    ax.set_ylabel("Sample $\\kappa$")

    # Beta hist
    ax = axes[2]
    ax.hist(sample.betas, color=hist_colors["posterior"], bins=20, density=True, lw=0.5, alpha=0.5)
    if has_groundtruth:
        ax.axvline(groundtruth.beta, label="Ground truth", color=dark_reference, lw=2, ls="--")
    ax.axvline(mercator_embedding.beta, label="Mercator", color=alg_colors["mercator"], lw=2, ls="--")
    ax.set_xlabel("Sample $\\beta$")
    ax.get_yaxis().set_visible(False)
    ax.spines[["left"]].set_visible(False)

    # Legend
    ax.spines[["left"]].set_visible(False)
    ax.legend(list(map(lambda x: Patch(color=alg_colors[x]), ["posterior", "mercator", "reference"])),
              ["Posterior sample", "Mercator", "Ground truth"],
               handlelength=1.5
              )
    ax.get_yaxis().set_visible(False)

    output_dir = figure_dir()
    fig.savefig(str(output_dir.joinpath(f"error_bars_{config_name}.pdf")),
                    bbox_inches="tight")
    fig.tight_layout()
    pyplot.show()
