import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from tqdm import tqdm
import basegraph

from modelingtools import merge_chains, generate_graph
from plot_tools import column_width, alg_colors, hist_colors, figure_dir, dark_reference
from pybigue.models import S1Model
from pybigue.sampling import read_sample
from pybigue.embedding_info import EmbeddingParameters, GraphInformation
from paths import ResultPaths, read_dataclass, read_graph


percentiles = [25, 75]

def hist(ax, values, **kwargs):
    ax.hist(values, density=True, bins=20, **kwargs)
    ax.get_yaxis().set_visible(False)
    ax.spines["left"].set_visible(False)

def graph_vector_correlation(ax, values, reference_values, labels, cs):
    for values, label, color, shift in zip(values, labels, cs, np.linspace(0, 0.3, len(values))):
        values = np.asarray(values).T
        label_shown = False
        for generated_values, original_value in zip(values, reference_values):
            original_value += shift

            median = np.median(generated_values)
            low, high = np.percentile(generated_values, percentiles)
            ax.errorbar(original_value, median, yerr=[[low], [high]], markersize=2, color=color, label=label if not label_shown else None, marker="o")
            label_shown = True
    xvalues = [np.min(reference_values), np.max(reference_values)]
    ax.plot(xvalues, xvalues, ls="--", color=alg_colors["reference"], zorder=0)


def get_h2_coordinates(embedding):
    kappa_min = 1
    radii = S1Model.get_H2_radii(np.array(embedding.kappa), kappa_min, S1Model.get_mu(embedding.beta, graph_info.average_degree))
    return np.concatenate([radii, embedding.theta]).reshape(2, -1).T

def auc(g, embedding, average_degree):
    n = g.get_size()
    probs = [S1Model.get_edge_prob(embedding, average_degree, i, j) for i in range(n-1) for j in range(i+1, n)]
    edges = [g.has_edge(i, j) for i in range(n-1) for j in range(i+1, n)]

    return metrics.roc_auc_score(edges, probs)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config_file", help="Config file path")
    parser.add_argument("-i", "--init", default="random", help="Initialization", choices=["random", "groundtruth", "mercator"])
    parser.add_argument("--start", help="Skip first s iterations.", default=0, type=int)
    parser.add_argument("-m", help="Display one point every m sample point.", default=1, type=int)
    parser.add_argument("-N", help="Graphs generated from posterior", default=500, type=int)
    args = parser.parse_args()
    config_name = os.path.splitext(Path(args.config_file).name)[0]

    np.random.seed(37)

    path_manager = ResultPaths(config_name)
    if os.path.isfile(path_manager.groundtruth_embedding_path):
        groundtruth_embedding = read_dataclass(EmbeddingParameters, path_manager.groundtruth_embedding_path)
    else:
        groundtruth_embedding = None
    graph_info = read_dataclass(GraphInformation, path_manager.graph_information_path)

    n = graph_info.n
    graph = read_graph(path_manager.graph_path)

    func_on_h2 = lambda func: lambda g, embedding: func(g, get_h2_coordinates(embedding))
    embedding_properties = {
            # "Greedy stability": func_on_h2(lambda g, embedding: min(basegraph.geometry.get_greedy_stability(g, embedding)),
            "Global hierarchy level": func_on_h2(lambda g, embedding: basegraph.geometry.get_hierarchy_levels(g, embedding[:, 0], embedding[:, 1])),
            "Sample $\\beta$": lambda _, embedding: embedding.beta,
            # "AUC": lambda g, embedding: auc(g, embedding, graph_info.average_degree),
        }
    greedy_score_names = {
            "Greedy success rate": "successful",
            # "Stretch": "stretch",
            # "Hyperstretch greedy": "hyperstretch greedy",
            # "Hyperstretch original": "hyperstretch original",
        }
    def compute_embedding_properties(container, embedding, original_graph):
        greedy_scores = basegraph.geometry.get_greedy_routing_scores(original_graph, get_h2_coordinates(embedding))
        for property_name, _ in greedy_score_names.items():
            container[property_name].append(greedy_scores[greedy_score_names[property_name]])
        for property_name, func in embedding_properties.items():
            container[property_name].append(func(original_graph, embedding))


    graph_properties = {
            "Density": lambda g: g.get_edge_number()*2/(g.get_size()*(g.get_size()-1)),
            "Clustering": lambda g: basegraph.metrics.get_global_clustering_coefficient(g),
            "degrees": lambda g: g.get_degrees(),
        }
    def compute_graph_properties(container, embedding, average_degree):
        posterior_graph = generate_graph(embedding, average_degree)
        for property_name, func in graph_properties.items():
            container[property_name].append(func(posterior_graph))


    plot_types = {key: "geom" for key in greedy_score_names} \
            | {
                "Greedy stability": "geom",
                "AUC": "geom",
                "Global hierarchy level": "geom",
                "Density": "graph scalar",
                "Clustering": "graph scalar",
                "degrees": "graph vector",
                "Sample $\\beta$": "geom"
            }

    all_properties = {name: [] for name in (graph_properties|greedy_score_names|embedding_properties).keys() }
    init = args.init
    path_manager.sample_prefix = init
    chain_samples = merge_chains(read_sample(path_manager.sample_dir_path))
    if len(chain_samples) == 0:
        exit()

    inferred_parameters = [param for param, sample in chain_samples.items() if sample is not None]
    sample_size = len(chain_samples)
    for i in tqdm(range(args.start, sample_size, args.m), total=(sample_size-args.start)//args.m):
        embedding = chain_samples[i]
        compute_embedding_properties(all_properties, embedding, graph)
        for _ in range(args.N):
            compute_graph_properties(all_properties, embedding, graph_info.average_degree)

    original_properties = {name: [] for name in all_properties.keys()}
    if groundtruth_embedding is not None:
        compute_embedding_properties(original_properties, groundtruth_embedding, graph)
    for property_name, func in graph_properties.items():
        original_properties[property_name].append(func(graph))

    if os.path.isfile(path_manager.mercator_embedding_path):
        mercator_embedding = read_dataclass(EmbeddingParameters, path_manager.mercator_embedding_path)
        mercator_properties = {name: [] for name in all_properties.keys()}
        compute_embedding_properties(mercator_properties, mercator_embedding, graph)
        for i in range(2000):
            compute_graph_properties(mercator_properties, mercator_embedding, graph_info.average_degree)
    else:
        mercator_properties = {}

    cols = 2
    rows = int(np.ceil(len(all_properties) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(column_width, column_width/2.5*rows))

    label = "Posterior sample"
    mercator_label = "Mercator embedding"

    for ax in axes.ravel():
        ax.set_axis_off()

    # for (property_name, values), ax in zip(all_properties.items(), axes.ravel()):
    # Specific for the paper
    for (property_name, values), ax in zip([(key, all_properties[key]) for key in ["Sample $\\beta$", "Density", "Clustering", "Greedy success rate", "Global hierarchy level"]], axes.ravel()):
        original_values = original_properties[property_name]
        mercator_values = mercator_properties.get(property_name)
        data = list(filter(lambda x: x is not None, [values, mercator_values]))
        colors = [hist_colors["posterior"], hist_colors["mercator"]][:len(data)]
        labels = [label, mercator_label][:len(data)]
        ax.set_axis_on()
        match plot_types[property_name]:
            case "graph scalar":
                hist(ax, data, label=labels, color=colors)
                if original_values != []:
                    ax.axvline(original_values[0], label="Original graph", lw=2.5, ls="--", color=dark_reference)
                ax.set_xlabel(property_name)
            case "geom":
                hist(ax, values, label=label, color=hist_colors["posterior"], alpha=0.5)
                ax.axvline(np.median(values), lw=2.5, ls="--", color=alg_colors["posterior"])
                if original_values != []:
                    ax.axvline(original_values[0], label="Original graph", lw=2.5, ls="--", color=dark_reference)
                if mercator_values is not None:
                    ax.axvline(mercator_values, label=mercator_label, lw=2.5, ls="--", color=alg_colors["mercator"])
                ax.set_xlabel(property_name)
            case "graph vector":
                if original_values != []:
                    graph_vector_correlation(ax, data, original_values[0], labels=labels, cs=colors)
                ax.set_ylabel("Predicted " + property_name)
                ax.set_xlabel("Original " + property_name)
    fig.tight_layout()

    output_dir = figure_dir()
    fig.savefig(str(output_dir.joinpath(f"properties_{config_name}.pdf")),
                    bbox_inches="tight")
    fig.tight_layout()
    plt.show()
