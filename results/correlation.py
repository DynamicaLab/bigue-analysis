import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import pingouin as pg
from tqdm import tqdm

from modelingtools import merge_chains, read_adjusted_sample
from paths import ResultPaths, read_dataclass, read_graph
from plot_tools import figure_dir
from pybigue.embedding_info import EmbeddingParameters, GraphInformation
from pybigue.metrics import circular_average
from pybigue.utils import angle_modulo


def grid_from_xy(xs, ys):
    xx, yy = np.meshgrid(xs, ys)
    return np.append(xx.reshape(-1,1), yy.reshape(-1,1), axis=1)


def kde(ax, positions, xperiodic, yperiodic, points=100):
    xmin, xmax = np.min(positions[0]), np.max(positions[0])
    ymin, ymax = np.min(positions[1]), np.max(positions[1])

    x_flat = np.linspace(xmin, xmax, points)
    y_flat = np.linspace(ymin, ymax, points)

    kernel = stats.gaussian_kde(positions, bw_method=0.3)
    z = np.zeros(points*points)
    for xshift in ([-2*np.pi, 0, 2*np.pi] if xperiodic else [0]):
        for yshift in ([-2*np.pi, 0, 2*np.pi] if yperiodic else [0]):
            z += kernel(grid_from_xy(x_flat+xshift, y_flat+yshift).T)
    z = z.reshape(points, points)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    xx, yy = np.meshgrid(x_flat, y_flat)
    ax.contourf(xx, yy, z, cmap='rocket')
    ax.contour(xx, yy, z, colors='k', linewidths=.7)


def interpolate_pvalue(energy, centile_values):
    if np.isnan(energy):
        return np.nan
    if energy>=np.max(centile_values):
        return 0
    if energy<=np.min(centile_values):
        return 1

    diff = energy-centile_values
    i = np.argmin(np.abs(diff))
    inf, sup = (i, i+1) if diff[i] > 0 else (i-1, i)
    slope = (centile_values[sup]-centile_values[inf])/.5
    centiles = np.arange(0, 100.5, 0.5)
    centile = (energy-centile_values[inf])/slope + centiles[inf]
    return 1-centile/100

def param_name(i, n):
    if i < n:
        return f"$\\theta_{{{i+1}}}$"
    elif i < 2*n:
        return f"$\\kappa_{{{i-n+1}}}$"
    return "$\\beta$"


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config_file", help="Config file path")
    parser.add_argument("-i", "--init", default="random", help="Initialization", choices=["random", "groundtruth", "mercator"])
    parser.add_argument("--all", action="store_true", help="Save large png containing every marginal posterior for pairs of angles.")
    args = parser.parse_args()
    config_name = os.path.splitext(Path(args.config_file).name)[0]

    path_manager = ResultPaths(config_name)
    path_manager.sample_prefix = args.init

    if not os.path.isfile(path_manager.groundtruth_embedding_path):
        theta_reference = np.asarray(read_dataclass(EmbeddingParameters, path_manager.mercator_embedding_path).theta)
    else:
        theta_reference = np.asarray(read_dataclass(EmbeddingParameters, path_manager.groundtruth_embedding_path).theta)

    graph = read_graph(path_manager.graph_path)
    graph_info = read_dataclass(GraphInformation, path_manager.graph_information_path)

    all_samples = merge_chains(read_adjusted_sample(path_manager, theta_reference))
    vals = [np.array(all_samples.thetas), np.array(all_samples.kappas), np.array([all_samples.betas]).T]
    samples = np.concatenate(vals, axis=1)

    mercator_parameters = read_dataclass(EmbeddingParameters, path_manager.mercator_embedding_path)

    n = graph_info.n
    sample_size = len(samples)
    n_parameters = samples.shape[1]
    n_parameters = int((samples.shape[1]-1)/2) # ignore kappas and beta

    df = {"p1": [], "p2": [], "p-value": []}
    max_p = (0, -1, -1)

    data = []
    for i in tqdm(range(n_parameters-1)):
        param_i = samples[:, i]
        if i < n:
            param_i = angle_modulo(param_i-circular_average(param_i))
        data.append(param_i)
        for j in tqdm(range(i+1, n_parameters), leave=False):

            if graph_info.fixed_vertices[0] in [i, j]:
                pvalue = np.nan
            else:
                param_j = samples[:, j]
                if i < n:
                    param_j = angle_modulo(param_j-circular_average(param_j))

                pvalue = pg.multivariate_normality(np.array([param_i, param_j]).T).pval
                if pvalue > max_p[0]:
                    max_p = pvalue, i, j
            df["p1"].append(param_name(i, graph_info.n))
            df["p2"].append(param_name(j, graph_info.n))
            df["p-value"].append(pvalue)
    print("global p-value", pg.multivariate_normality(np.array(data).T).pval)
    print(max_p)

    g = sns.relplot(
        data=pd.DataFrame.from_dict(df),
        x="p1", y="p2", hue="p-value", size="p-value",
        palette="vlag", hue_norm=(0, 0.05), edgecolor="#383838",
        height=6, sizes=(20, 200), size_norm=(0, 1),
        zorder=5,
    )
    # g.ax.grid(True)
    plt.xlabel("")
    plt.ylabel("")
    plt.gca().spines[["bottom", "left"]].set_visible(False)
    plt.gca().tick_params(axis=u'both', which=u'both',length=0)
    plt.tight_layout()
    plt.title("$\\theta$ pairwise marginal p-values")
    plt.show()

    output_dir = figure_dir()
    # Large pair marginals
    if args.all:
        fig, axes = plt.subplots(n_parameters, n_parameters, figsize=(n_parameters, n_parameters))
        for i in tqdm(range(n_parameters)):
            for j in tqdm(range(n_parameters), leave=False):
                ax = axes[i, j]
                if i>=j or i==graph_info.fixed_vertices[0] or j==graph_info.fixed_vertices[0]:
                    fig.delaxes(ax)
                    continue
                xs = samples[:, i]
                ys = samples[:, j]
                if i < n:
                    xs = angle_modulo(xs-circular_average(xs))
                if j < n:
                    ys = angle_modulo(ys-circular_average(ys))

                kde(ax, np.stack([xs, ys]), i<n, j<n)
                # ax.scatter(xs, ys, alpha=0.2, s=8)
                if i == j-1:
                    ax.set_xlabel(param_name(i, n))
                    ax.set_ylabel(param_name(j, n))

                ax.spines[["bottom", "left"]].set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])
        fig.tight_layout()
        output_dir = figure_dir()
        fig.savefig(str(output_dir.joinpath(f"correlations_{config_name}.png")), dpi=400)
