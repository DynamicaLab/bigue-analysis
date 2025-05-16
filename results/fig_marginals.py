import os
import string

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import lines
from scipy import stats

from paths import ResultPaths, read_dataclass
from pybigue.embedding_info import EmbeddingParameters, GraphInformation

from modelingtools import merge_chains, read_adjusted_sample
from plot_tools import figure_dir, format_ticks, column_width
from pybigue.metrics import circular_average
from pybigue.utils import angle_modulo


def grid_from_xy(xs, ys):
    xx, yy = np.meshgrid(xs, ys)
    return np.append(xx.reshape(-1,1), yy.reshape(-1,1), axis=1)


def kde(ax, positions, xperiodic, yperiodic, points=100):
    xymin, xymax = np.min(positions), np.max(positions)

    x_flat = np.linspace(xymin, xymax, points)
    y_flat = np.linspace(xymin, xymax, points)

    kernel = stats.gaussian_kde(positions.T, bw_method=0.3)
    z = np.zeros(points*points)
    for xshift in ([-2*np.pi, 0, 2*np.pi] if xperiodic else [0]):
        for yshift in ([-2*np.pi, 0, 2*np.pi] if yperiodic else [0]):
            z += kernel(grid_from_xy(x_flat+xshift, y_flat+yshift).T)
    z = z.reshape(points, points)

    ax.set_xlim(xymin, xymax)
    ax.set_ylim(xymin, xymax)

    xx, yy = np.meshgrid(x_flat, y_flat)
    ax.contour(xx, yy, z, colors='k', linewidths=.7)


def param_name(i, n):
    if i < n:
        ind = f"{i+1}"
        return f"$\\theta_{{{ind}}}-\\bar{{\\theta}}_{{{ind}}}$"
    elif i < 2*n:
        return f"$\\kappa_{{{i-n+1}}}$"
    return "$\\beta$"


if __name__ == "__main__":
    fig, axes = plt.subplots(2, 2, figsize=(column_width, column_width))

    fig.delaxes(axes[1, 1])
    colors = ["#EEAAD7", "#9567E0", "#89B8D2", "#DFA953"]
    for id, ((config_name, (v1, v2)), ax) in enumerate(zip([("30v_long", (4, 7)), ("karate", (17, 19)), ("conflicting", (7, 15))], axes.flatten())):
        path_manager = ResultPaths(config_name)
        path_manager.sample_prefix = "random"

        if not os.path.isfile(path_manager.groundtruth_embedding_path):
            theta_reference = np.asarray(read_dataclass(EmbeddingParameters, path_manager.mercator_embedding_path).theta)
        else:
            theta_reference = np.asarray(read_dataclass(EmbeddingParameters, path_manager.groundtruth_embedding_path).theta)

        graph_info = read_dataclass(GraphInformation, path_manager.graph_information_path)

        n = graph_info.n

        output_dir = figure_dir()
        chain_samples = read_adjusted_sample(path_manager, theta_reference)

        all_theta = np.array(merge_chains(chain_samples).thetas)
        xs = all_theta[:, v1]
        ys = all_theta[:, v2]
        x_average = circular_average(xs)
        y_average = circular_average(ys)
        xs = angle_modulo(xs - x_average)
        ys = angle_modulo(ys - y_average)
        kde(ax, np.stack([xs, ys]).T, v1<n, v2<n)

        for sample, color in zip(chain_samples.values(), colors):
            xs = angle_modulo(sample.thetas[:, v1] - x_average)
            ys = angle_modulo(sample.thetas[:, v2] - y_average)
            ax.scatter(xs, ys, alpha=.3, s=3, color=color, edgecolor="none")

        ax.set_xlabel(param_name(v1, n))
        ax.set_ylabel(param_name(v2, n), labelpad=-0.2)
        ax.set_xticks(*format_ticks(1))
        ax.set_yticks(*format_ticks(1))
        ax.text(-0.3, 1.0, f"\\textbf{{{string.ascii_lowercase[id]}}}", transform=ax.transAxes,
                    size="large")

    fig.legend([lines.Line2D([], [], ls="none", marker=".", color=c) for c in colors],
               [f"Chain {i+1}" for i in range(4)],
               loc='center', bbox_to_anchor=(0.75, .25))
    fig.subplots_adjust(top=.95, right=0.95, wspace=0.5, hspace=0.5)

    output_dir = figure_dir()
    fig.savefig(str(output_dir.joinpath(f"fig_marginals.pdf")))
    plt.show()
