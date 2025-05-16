import numpy as np
from matplotlib import pyplot

from paths import ResultPaths, read_dataclass, read_graph
from pybigue.embedding_info import EmbeddingParameters, GraphInformation
from pybigue.metrics import circ_seff_rhat, reg_seff_rhat
from pybigue.models import S1Model
from pybigue.sampling import read_sample

from plot_tools import figure_dir, column_width, alg_names, alg_colors, darken_color, all_colors
from modelingtools import get_chain_loglikelihoods


def plot_with_rhat(ax, loglikelihoods, config_name, color, xpos, warmup=0, zorder=None, prec=3, unthin=1):
    ax.plot(unthin*np.arange(len(loglikelihoods)), loglikelihoods,
            label=alg_names[config_name],
            color=color, alpha=0.7, lw=1,
            zorder=zorder
            )
    convergence_stats = \
            [circ_seff_rhat([np.array(sample.thetas)[warmup:, v] for sample in chain_samples.values()]) for v in graph if v!=graph_info.fixed_vertices[0]]\
            + [reg_seff_rhat([np.array(sample.kappas)[warmup:, v] for sample in chain_samples.values()]) for v in graph]\
            + [reg_seff_rhat([sample.betas[warmup:] for sample in chain_samples.values()])]
    summary_rhat = max(rhat for _, rhat in convergence_stats)
    seffs = list(seff for seff, _ in convergence_stats)
    low, high = np.percentile(seffs, [25, 75])
    print(alg_names[config_name], f": {np.median(seffs):.0f} [{low:.0f}, {high:.0f}]")

    ax.text(xpos, 0.18,
            f"\n$\\hat{{R}}_\\text{{max}}={summary_rhat:.{prec}f}$",
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes,
            color= darken_color(color, light_correction=0.7 if color==all_colors[4] else 0.9, saturation_correction=2)
            )


if __name__ == "__main__":
    path_manager = ResultPaths("30v") # Arbitary, used to obtain graph and graph_info
    graph = read_graph(path_manager.graph_path)
    graph_info = read_dataclass(GraphInformation, path_manager.graph_information_path)
    groundtruth_parameters = read_dataclass(EmbeddingParameters, path_manager.groundtruth_embedding_path)

    adjacency = np.array(graph.get_adjacency_matrix(False))
    ground_truth_likelihood = S1Model.loglikelihood(adjacency, graph_info.average_degree, **groundtruth_parameters.as_dict())


    fig, axes = pyplot.subplots(2, figsize=(column_width, 0.9*column_width), sharex=True)
    fig2, ax2 = pyplot.subplots(1, figsize=(column_width, 0.43*column_width))

    colors = [alg_colors["posterior"], alg_colors["rw"], alg_colors["hmc"]]
    zorder = 4
    warmup = 6000
    chain = "0"
    for i, (config_name, color) in enumerate(zip(["30v", "30v_rw", "30v_hmc"], colors)):
        path_manager = ResultPaths(config_name)
        for init, ax in zip(["random", "groundtruth"], axes):
            path_manager.sample_prefix = init

            chain_samples = read_sample(path_manager.sample_dir_path)

            loglikelihoods = get_chain_loglikelihoods(graph, chain_samples[chain], chain, EmbeddingParameters(), path_manager)
            print(f"({init})", end=" ")
            plot_with_rhat(ax, loglikelihoods, config_name, color, xpos=1/6+i/3, warmup=warmup, zorder=zorder, unthin=10)
            zorder -= 1

    for i, (config_name, color) in enumerate(zip(["30v_long", "30v_rw_long"], colors)):
        path_manager = ResultPaths(config_name)
        path_manager.sample_prefix = "groundtruth"
        chain_samples = read_sample(path_manager.sample_dir_path)
        print(config_name, len(list(chain_samples.values())[0]))
        loglikelihoods = [ground_truth_likelihood]+get_chain_loglikelihoods(graph, chain_samples[chain], chain, EmbeddingParameters(), path_manager).tolist()
        plot_with_rhat(ax2, loglikelihoods, config_name, color, xpos=1/4+i/2, prec=5)

    mercator_loglikelihood = S1Model.loglikelihood(
            adjacency, graph_info.average_degree,
            **read_dataclass(EmbeddingParameters, path_manager.mercator_embedding_path).as_dict()
        )

    for ax in list(iter(axes)) + [ax2]:
        ax.axhline(mercator_loglikelihood, color=alg_colors["mercator"], lw=1.5, label="Mercator")
        ymin, ymax = ax.get_ylim()
        ax.set_ylim((ymin*1.03, ymax))
        ax.set_ylim(-300, -90)
        ax.set_ylabel("Log-likelihood")
        ax.get_yaxis().set_label_coords(-0.16, 0.5)
        ax.ticklabel_format(axis='y')
        ax.axhline(ground_truth_likelihood, label="Ground truth", color="k", lw=1.5, zorder=0)

    for ax in axes:
        ymin, ymax = ax.get_ylim()
        ax.set_xlim((-1000, 200000))
        ax.fill_between([-1000, 10*warmup], [1.1*ymin]*2, [ymax]*2, color="#eaeaea", zorder=-10)

    ax2.axhline(mercator_loglikelihood, color=alg_colors["mercator"], lw=1.5)
    ax2.set_xlabel("Iterations (thinned)")
    ax2.set_xlim(-1, 501)
    axes[-1].set_xlabel("Iterations")
    fig.tight_layout()
    fig2.tight_layout()

    fig.subplots_adjust(left=0.19, bottom=0.15, right=0.94, top=0.75, wspace=0, hspace=0.2)
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.13), ncols=2)

    output_dir = figure_dir()
    fig.savefig(str(output_dir.joinpath("alg_comparisons_30v.pdf")))
    fig2.savefig(str(output_dir.joinpath("alg_comparisons_30v_long.pdf")))
    pyplot.show()
