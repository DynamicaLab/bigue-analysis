import os

import numpy as np
from sklearn import metrics
from tqdm import tqdm
import basegraph
import arviz as az
from matplotlib import pyplot as plt

from pybigue.embedding_info import EmbeddingParameters, EmbeddingsContainer, GraphInformation
from pybigue.metrics import circ_seff_rhat, reg_seff_rhat
from pybigue.models import S1Model
from pybigue.sampling import read_sample

from modelingtools import merge_chains, generate_graph
from paths import ResultPaths, read_dataclass, read_graph
from plot_tools import column_width, alg_colors, figure_dir, dark_reference


def get_h2_coordinates(embedding):
    kappa_min = 1
    radii = S1Model.get_H2_radii(np.array(embedding.kappa), kappa_min, S1Model.get_mu(embedding.beta, graph_info.average_degree))
    return np.concatenate([radii, embedding.theta]).reshape(2, -1).T

def auc(g, embedding, average_degree):
    n = g.get_size()
    probs = [S1Model.get_edge_prob(embedding, average_degree, i, j) for i in range(n-1) for j in range(i+1, n)]
    edges = [g.has_edge(i, j) for i in range(n-1) for j in range(i+1, n)]

    return metrics.roc_auc_score(edges, probs)

def plot_bars(ax, prop, ypos, lw=None, s=None, marker=".", **kwargs):
    if len(prop) > 1:
        low, high = az.hdi(np.array(prop), hdi_prob=0.5)
        ax.plot([low, high], [ypos, ypos], **kwargs)
        ax.scatter(np.median(prop), ypos, lw=lw, marker=marker, s=s, **kwargs)
    elif len(prop) == 1:
        ax.scatter(prop, ypos, lw=lw, marker=marker, s=s, **kwargs)


def format_table(prop, fmt, quartiles=False):
    if len(prop) > 1:
        low, high = np.percentile(prop, [25, 75]) if quartiles else az.hdi(np.array(prop), hdi_prob=0.5)
        return f"{np.median(prop):{fmt}} [{low:{fmt}}, {high:{fmt}}]"
    elif len(prop) == 1:
        return f"{np.median(prop):{fmt}} [---]"
    return "---"


datasets = {
        "karate": "Zachary",
        "macaque_neural": "Macaque",
        "montreal": "Gangs",
        "dutch_criticism": "Critics",
        "zebras": "Zebras",
        "november17": "Terrorism",
        "kangaroo": "Kangaroo",
        "new_guinea_tribes": "Tribes",
}

if __name__ == "__main__":
    np.random.seed(37)
    n_posterior_pred = 100
    init = "random"
    cols = ["Density", "Clustering", "Shortest path average\nlength", "Global hierarchy level"]

    fig, axes = plt.subplots(len(datasets), len(cols), figsize=(2*column_width, (.1*len(datasets)+.2)*column_width), sharex="col")

    table_output = "Dataset &" + " & ".join([r"$|V|$", r"$S_\text{eff}$"] + cols)+"\\\\\n"
    if axes.ndim == 1:
        axes = np.array([axes])
    for row, (config_name, ax_row) in enumerate(zip(datasets, axes)):
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
                "AUC ROC": lambda g, embedding: auc(g, embedding, graph_info.average_degree),
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
                # "Number of edges":
                "Density": lambda g: g.get_edge_number()*2/(g.get_size()*(g.get_size()-1)),
                "Clustering": lambda g: basegraph.metrics.get_global_clustering_coefficient(g),
                "Shortest path average\nlength": lambda g: np.average(basegraph.metrics.get_shortest_path_averages(g))
                # "degrees": lambda g: g.get_degrees(),
            }
        def compute_graph_properties(container, embedding, average_degree):
            posterior_graph = generate_graph(embedding, average_degree)
            for property_name, func in graph_properties.items():
                container[property_name].append(func(posterior_graph))


        all_properties = {name: [] for name in (graph_properties|greedy_score_names|embedding_properties).keys() }
        path_manager.sample_prefix = init
        chain_samples = {
                chain: EmbeddingsContainer(sample.thetas[:300], sample.kappas[:300], sample.betas[:300])
                for chain, sample in read_sample(path_manager.sample_dir_path).items()}
        all_samples = merge_chains(chain_samples)

        if len(all_samples) == 0:
            print(f"graph {config_name} not sampled")
            continue

        if not os.path.isfile(path_manager.mercator_embedding_path):
            print(f"Mercator not ran for {config_name}")
            continue

        mercator_embedding = read_dataclass(EmbeddingParameters, path_manager.mercator_embedding_path)
        mercator_properties = {name: [] for name in all_properties.keys()}
        compute_embedding_properties(mercator_properties, mercator_embedding, graph)
        for i in range(2000):
            compute_graph_properties(mercator_properties, mercator_embedding, graph_info.average_degree)

        inferred_parameters = [param for param, sample in all_samples.items() if sample is not None]
        sample_size = len(all_samples)
        for i in tqdm(range(sample_size)):
            embedding = all_samples[i]
            compute_embedding_properties(all_properties, embedding, graph)
            for _ in range(n_posterior_pred):
                compute_graph_properties(all_properties, embedding, graph_info.average_degree)

        original_properties = {name: [] for name in all_properties.keys()}
        if groundtruth_embedding is not None:
            compute_embedding_properties(original_properties, groundtruth_embedding, graph)
        for property_name, func in graph_properties.items():
            original_properties[property_name].append(func(graph))

        seffs = \
                [circ_seff_rhat([np.array(sample.thetas)[:, v] for sample in chain_samples.values()])[0] for v in graph if v!=graph_info.fixed_vertices[0]]\
                + [reg_seff_rhat([np.array(sample.kappas)[:, v] for sample in chain_samples.values()])[0] for v in graph]\
                + [reg_seff_rhat([sample.betas for sample in chain_samples.values()])[0]]

        table_output += datasets[config_name] + " & "
        table_output += str(graph.get_size()) + " & "
        table_output += format_table(seffs, ".0f", quartiles=True) + " & "

        for ax, property in zip(ax_row, cols):
            dist = 0.75
            plot_bars(ax, all_properties[property], -dist, color=alg_colors["posterior"])
            plot_bars(ax, mercator_properties[property], 0, marker="x", s=14, lw=.7, color=alg_colors["mercator"], zorder=4)
            plot_bars(ax, original_properties[property], dist, marker="*", s=28, lw=0, color=dark_reference, zorder=5)

            ax.spines[['left']].set_visible(False)
            ax.set_yticks([])
            ax.set_ylim(-1.5, 1.5)

            if cols.index(property) == 0:
                ax.set_ylabel(datasets[config_name], rotation=0, labelpad=20)
            else:
                ax.get_yaxis().set_visible(False)
            if row < len(datasets)-1:
                ax.get_xaxis().set_visible(False)
                ax.spines[['bottom']].set_visible(False)
            else:
                ax.set_xlabel(property)

            fmt = ".2f"
            gt_output = "---" if len(original_properties[property]) == 0 else\
                        f"{original_properties[property][0]:{fmt}}"
            table_output += gt_output + ", & "\
                            + format_table(all_properties[property], fmt) + ", & "\
                            + format_table(mercator_properties[property], fmt)

            table_output += " & " if property!=cols[-1] else r"\\"
        table_output += "\n"
    print(table_output)

    fig.tight_layout()
    plt.subplots_adjust(hspace=.5, wspace=0.15)
    output_dir = figure_dir()
    fig.savefig(str(output_dir.joinpath("empirical_networks.pdf")))
    with open(str(output_dir.joinpath("empirical_networks_values.txt")), "w") as file_stream:
        file_stream.write(table_output)
    # plt.show()
