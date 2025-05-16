import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from paths import ResultPaths, read_dataclass, read_graph
from pybigue.sampling import read_sample
from pybigue.embedding_info import EmbeddingParameters, GraphInformation
from pybigue.metrics import circ_seff_rhat, reg_seff_rhat, normalized_circ_autocovariance, normalized_reg_autocovariance
from plot_tools import column_width, figure_dir


init_full_names = {
    "random": "Random initialization",
    "mercator": "Mercator initialization",
    "groundtruth": "Ground truth initialization"
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config_file", help="Config file path")
    parser.add_argument("-i", "--init", nargs="+",
                        default=["random"],
                        help="Initialization", choices=["random", "groundtruth", "mercator"])
    parser.add_argument("--start", "-s", help="Skip first s iterations.",
                        default=0, type=int)
    parser.add_argument("-m", help="Display one point every m sample point.",
                        default=1, type=int)
    parser.add_argument("--chain", "-c", help="Display only chain of given name",
                        default=None, type=str)
    args = parser.parse_args()

    config_name = os.path.splitext(Path(args.config_file).name)[0]

    path_manager = ResultPaths(config_name)

    graph = read_graph(path_manager.graph_path)
    graph_info = read_dataclass(GraphInformation, path_manager.graph_information_path)
    mercator_parameters = read_dataclass(EmbeddingParameters, path_manager.mercator_embedding_path)

    inits = args.init
    if len(inits) != 1:
        fig, axes = plt.subplots(len(inits), 1, figsize=(5, 2*len(inits)), sharey=True, sharex=True)
    else:
        fig, axes = plt.subplots(len(inits), 1, figsize=(column_width, 0.5*column_width), sharey=True, sharex=True)
        axes = [axes]

    reference_theta = np.array(mercator_parameters.theta)
    sorted_index = np.argsort(reference_theta)

    lags = np.arange(0, 21)


    colors = ["#EEAAD7", "#9567E0", "#89B8D2", "#DFA953"]
    for init, ax in zip(inits, axes):
        path_manager.sample_prefix = init
        if not os.path.isdir(path_manager.sample_dir_path):
            continue
        chain_samples = read_sample(path_manager.sample_dir_path)
        if chain_samples == {}:
            continue

        if len(inits) != 1:
            ax.set_title(init_full_names[init], size=10)
        for i, (chain, samples) in enumerate(chain_samples.items()):
            if args.chain is not None and args.chain != chain:
                continue
            theta = np.array(samples.thetas)[args.start::args.m]
            kappa = np.array(samples.kappas)[args.start::args.m]
            beta = np.array(samples.betas)[args.start::args.m]

            av_autocovariances = np.average(
                    [[normalized_circ_autocovariance(np.array(theta)[:, v], lag) for lag in lags] for v in graph if v!=graph_info.fixed_vertices[0]]
                    + [[normalized_reg_autocovariance(np.array(kappa)[:, v], lag) for lag in lags] for v in graph]
                    + [[normalized_reg_autocovariance(np.array(beta)[:], lag) for lag in lags]]
                   , axis=0)

            ax.plot(lags, av_autocovariances, label=f"Chain {int(chain)+1}", color=colors[i] if len(chain_samples)==4 else None)

        ax.axhline(0, ls="--", color="gray", zorder=0)

        convergence_stats = \
                [circ_seff_rhat([np.array(sample.thetas)[:, v] for sample in chain_samples.values()]) for v in graph if v!=graph_info.fixed_vertices[0]]\
                + [reg_seff_rhat([np.array(sample.kappas)[:, v] for sample in chain_samples.values()]) for v in graph]\
                + [reg_seff_rhat([sample.betas for sample in chain_samples.values()])]
        rhats = [rhat for _, rhat in convergence_stats]
        seffs = [seff for seff, _ in convergence_stats]

        print(f"max rhat of {init}", np.max(rhats))
        print(f"median seff of {init}", np.median(seffs))

        axes[-1].set_ylabel("Normalized\nautocovariance")
    axes[-1].set_xlabel("Lags")
    axes[-1].set_xlim((np.min(lags), np.max(lags)))
    fig.tight_layout()

    if len(inits) == 1:
        handles, labels = plt.gca().get_legend_handles_labels()
        if len(handles) == 4:
            order = [0, 3, 1, 2]
            plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], ncols=2)

        output_dir = figure_dir()
        fig.savefig(str(output_dir.joinpath(f"autocovariance_{config_name}.pdf")),
                    bbox_inches="tight")
    plt.show()
