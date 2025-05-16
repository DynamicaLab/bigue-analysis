import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.resolve()))
from config import parse_config

import os
from argparse import ArgumentParser
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

from paths import ResultPaths
from plot_tools import darken_color, hist_colors, column_width, figure_dir


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config_file", help="Config file path")
    args = parser.parse_args()
    config = parse_config(args.config_file)
    config_name = config["name"]

    path_manager = ResultPaths(config_name)

    n_bins = 9
    bins = np.linspace(0, 1, n_bins+1)

    # Reading output of simulations
    bigue_auc = []
    bigue_rank_average = []
    bigue_sample_average = []
    mercator_ranks = []
    mercator_auc = []
    gt_ranks = []
    for i in range(21):
        mercator_auc_output = os.path.join(path_manager.cache_dir_path, f"auc_mercator_{i}.txt")
        bigue_auc_output = os.path.join(path_manager.cache_dir_path, f"auc_bigue_{i}.txt")
        bigue_rank_average_output = os.path.join(path_manager.cache_dir_path, f"rank_average_bigue_{i}.txt")
        bigue_sample_average_output = os.path.join(path_manager.cache_dir_path, f"rank_sample_average_bigue_{i}.txt")

        mercator_rank_output = os.path.join(path_manager.cache_dir_path, f"ranks_mercator_{i}.txt")
        gt_output = os.path.join(path_manager.cache_dir_path, f"ranks_groundtruth_{i}.txt")

        if not os.path.isfile(bigue_rank_average_output):
            print(f"simulation {i} missing")
            continue
        bigue_auc.append(np.loadtxt(bigue_auc_output))
        bigue_rank_average.append(np.histogram(np.loadtxt(bigue_rank_average_output), bins=bins)[0])
        bigue_sample_average.append(np.histogram(np.loadtxt(bigue_sample_average_output), bins=bins)[0])
        mercator_ranks.append(np.histogram(np.loadtxt(mercator_rank_output), bins=bins)[0])
        mercator_auc.append(np.loadtxt(mercator_auc_output))
        if os.path.isfile(gt_output):
            gt_ranks.append(np.histogram(np.loadtxt(gt_output), bins=bins)[0])

    # AUC with with removed edges
    colors = [hist_colors["posterior"], hist_colors["mercator"]]
    fig, ax = plt.subplots(1, figsize=(.5*column_width, column_width/2.5), tight_layout=True)
    ax.hist([np.ravel(np.array(bigue_auc)), np.ravel(np.array(mercator_auc))], color=colors, density=True)
    ax.set_xlabel("AUC")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    fig_path = figure_dir()
    fig.savefig(str(fig_path.joinpath(f"auc_link_pred_{config_name}.pdf")),
                bbox_inches="tight")

    # Normalized ranks
    fig, ax = plt.subplots(1, figsize=(.5*column_width, column_width/2.5), tight_layout=True)
    res = [bigue_sample_average, mercator_ranks]
    colors = [hist_colors["posterior"], hist_colors["mercator"]]
    if gt_ranks != []:
        res.append(gt_ranks)
        colors.append(hist_colors["reference"])


    n_data = len(res)
    bin_widths = bins[1:]-bins[:-1]
    bin_centers = 0.5*(bins[1:]+bins[:-1])
    bin_scale = 0.8
    for i, (data, color) in enumerate(zip(res, colors)):
        q1, median, q3 = np.percentile(data, [25, 50, 75], axis=0)
        shift = (i/n_data-.5)*bin_widths*bin_scale
        ecolor = darken_color(color)
        plt.bar(bin_centers+shift, median, width=bin_scale*bin_widths/n_data, yerr=[median-q1, q3-median],
                color=color, error_kw={"ecolor": ecolor})

    ax.get_yaxis().set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Normalized rank")

    fig.savefig(str(fig_path.joinpath(f"ranks_{config_name}.pdf")),
                    bbox_inches="tight")
    plt.show()
