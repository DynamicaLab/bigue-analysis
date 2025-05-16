import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from tqdm import tqdm

import basegraph
from pybigue.embedding_info import EmbeddingParameters, GraphInformation, replace_known_parameters
from pybigue.kernels.transforms import get_global_sampling_kernel
from pybigue.models import S1Model
from pybigue.sampling import sample_chain, sample_hmc, sample_truncated_normal, sample_uniform
from pybigue.utils import align_theta, angle_modulo, gen_cauchy_lpdf, gen_normal_lpdf, sample_truncated_pareto

from config import parse_config
from generate_conflicting_embeddings import generate_conflicting
from util import embed_with_mercator
from results.paths import ResultPaths, write_graph
from results.modelingtools import get_auc, get_matrix_prob, generate_graph


def precision(edges, embedding, average_degree):
    return sum([S1Model.get_edge_prob(embedding, average_degree, i, j) > 0.5 for (i, j) in edges]) / len(edges)

def normalized_rank(edge_probs, n):
    """ Finds the average rank of removed edges among all non-edges of the graph
    (including removed edges). """
    return (edge_probs).argsort().argsort()[:n] / len(edge_probs)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("graph_id")
    parser.add_argument("--conflicting", action="store_true",
                        help="If flag unset, graphs are generated with the S^1 model. Otherwise, "
                             "a conflicting embedding is used.")
    parser.add_argument("-N", help="Number of graphs with randomly removed edges.", type=int, default=10)
    parser.add_argument("-p", help="Proportion of removed vertices.", type=float, default=0.05)
    parser.add_argument("--sample-size", help="Sample size.", type=int, default=20)
    parser.add_argument("--progress", help="Display progress bar.", action="store_true")
    args = parser.parse_args()


    config_dir = Path(__file__).resolve().parent.joinpath("config")
    config_file = "conflicting.yaml" if args.conflicting else "30v_long.yaml"
    config = parse_config(str(config_dir.joinpath(config_file)))
    path_manager = ResultPaths(config["name"])
    hyperparameters = config["hyperparameters"]
    alg_parameters = config["algorithm"]
    if seed := config.get("seed"):
        np.random.seed(seed)

    inferred_parameters = {"theta", "kappa", "beta"}
    known_parameters = EmbeddingParameters()

    np.random.seed(int(args.graph_id)) # Change graph and embedding for each simulation
    n = 30
    attempts = 100

    # Generate ground truth for graph_id
    groundtruth_parameters = None
    for attempt in range(attempts):
        groundtruth_parameters = EmbeddingParameters(
                    theta=sample_uniform(-np.pi, np.pi, n),
                    kappa=sample_truncated_pareto(4, 20, 3, n),
                    beta=2.5)
        if args.conflicting:
            theta2 = np.copy(groundtruth_parameters.theta)
            theta2[0] = angle_modulo(theta2[0] + 2)
            embedding2 = EmbeddingParameters(
                theta=theta2,
                kappa=groundtruth_parameters.kappa,
                beta=groundtruth_parameters.beta)

            original_graph = generate_conflicting(groundtruth_parameters, embedding2)
            groundtruth_parameters = None
        else:
            original_graph = generate_graph(
                        groundtruth_parameters,
                        np.average(groundtruth_parameters.kappa)
                    )
        components = basegraph.metrics.find_weakly_connected_components(original_graph)
        if len(components)==1:
            break
    else:
        raise RuntimeError(f"Coudn't generate a connected ground truth graph after {attempts} attempts.")

    edges = list(original_graph.edges())
    n_removed = int(round(args.p * original_graph.get_edge_number(), 0))
    all_non_edges = [(u, v) for u in range(original_graph.get_size()) for v in range(u) if not original_graph.has_edge(u, v)]
    path_manager.graph_file += "graph_link_pred.txt"
    write_graph(original_graph, path_manager.graph_path)


    if not os.path.isdir(path_manager.cache_dir_path):
        os.makedirs(path_manager.cache_dir_path)


    bigue_auc_stream = open(os.path.join(path_manager.cache_dir_path, f"auc_bigue_{args.graph_id}.txt"), "w")
    bigue_rank_average_stream = open(os.path.join(path_manager.cache_dir_path, f"rank_average_bigue_{args.graph_id}.txt"), "w")
    bigue_sample_average_stream = open(os.path.join(path_manager.cache_dir_path, f"rank_sample_average_bigue_{args.graph_id}.txt"), "w")
    mercator_rank_stream = open(os.path.join(path_manager.cache_dir_path, f"ranks_mercator_{args.graph_id}.txt"), "w")
    mercator_auc_stream = open(os.path.join(path_manager.cache_dir_path, f"auc_mercator_{args.graph_id}.txt"), "w")
    if groundtruth_parameters is not None:
        gt_rank_stream = open(os.path.join(path_manager.cache_dir_path, f"ranks_groundtruth_{args.graph_id}.txt"), "w")

    for it in (tqdm(range(args.N)) if args.progress else range(args.N)):
        suffix = f"_{args.graph_id}_{it}"
        if args.conflicting:
            suffix += "_conflicting"

        attempts = 100
        for attempt in range(attempts):
            graph = original_graph.get_deep_copy()
            removed_edges = list(map(lambda i: edges[i], np.random.choice(len(edges), n_removed, replace=False)))
            for edge in removed_edges:
                graph.remove_edge(*edge)
            components = basegraph.metrics.find_weakly_connected_components(graph)
            if len(components)==1:
                break
        else:
            raise RuntimeError(f"Coudn't generate a connected graph after {attempts} attempts.")
        all_non_edges_updated = removed_edges + all_non_edges

        graph_info = GraphInformation.from_degrees(graph.get_degrees())

        kappa_logprior = gen_cauchy_lpdf(0, hyperparameters.gamma) if "kappa" in inferred_parameters else lambda *_: 0
        beta_logprior = gen_normal_lpdf(hyperparameters.beta_average, hyperparameters.beta_std) if "beta" in inferred_parameters else lambda *_: 0
        adjacency = np.array(graph.get_adjacency_matrix(False))
        def logposterior(embedding):
            return S1Model.loglikelihood(adjacency, graph_info.average_degree, embedding.theta, embedding.kappa, embedding.beta)\
                    + np.sum(kappa_logprior(embedding.kappa)) + beta_logprior(embedding.beta)

        # Ground truth
        if groundtruth_parameters is not None:
            edge_probs = np.asarray([S1Model.get_edge_prob(groundtruth_parameters, graph_info.average_degree, i, j) for (i, j) in all_non_edges_updated])
            gt_ranks = normalized_rank(edge_probs, len(removed_edges))

        # Mercator
        path_manager = ResultPaths(config["name"])
        path_manager.graph_file = f"graph{suffix}.txt"
        write_graph(graph, path_manager.graph_path)
        path_manager.mercator_embedding_file = f"mercator_{config['name']}{suffix}.json"

        mercator_embedding = embed_with_mercator(path_manager, graph_info.fixed_vertices,
                                                             lambda x: S1Model.loglikelihood(adjacency, graph_info.average_degree, **x.as_dict()), verbose=False)

        edge_probs = np.asarray([S1Model.get_edge_prob(mercator_embedding, graph_info.average_degree, i, j) for (i, j) in all_non_edges_updated])
        mercator_ranks = normalized_rank(edge_probs, len(removed_edges))

        # Bigue
        def init(_): # random initialization
            embedding = replace_known_parameters(EmbeddingParameters(
                            theta=sample_uniform(-np.pi, np.pi, graph_info.n),
                            kappa=np.array([max(1e-10, deg) for deg in graph.get_degrees()]),
                            beta=sample_truncated_normal(hyperparameters.beta_average, hyperparameters.beta_std, size=1, lower_bound=1)),
                     known_parameters
            )
            embedding.theta = align_theta(embedding.theta, *graph_info.fixed_vertices)
            return embedding

        kernel = get_global_sampling_kernel(alg_parameters.kernels, init(0), known_parameters, graph, graph_info, hyperparameters, logposterior)
        sampling_args = {
                "kernel": kernel,
                "initial_embedding_generator": init,
                "sample_directory": None,
                "sample_size": args.sample_size,
                "thin": int(alg_parameters.thin),
            }
        if list(alg_parameters.kernels.keys()) == ["hmc"]:
            sample_mcmc = lambda chain_id, *_: sample_hmc(chain_id=chain_id, **sampling_args)
        else:
            sample_mcmc = lambda chain_id, log_progress: sample_chain(
                                warmup=alg_parameters.warmup, chain_id=chain_id, log_progress=log_progress, **sampling_args
                            )
        embeddings = sample_mcmc(0, lambda *_: None)

        probs = np.array([np.asarray([S1Model.get_edge_prob(embedding, graph_info.average_degree, i, j) for (i, j) in all_non_edges_updated])
                                  for embedding in embeddings])
        bigue_prob_average = normalized_rank(np.average(probs, axis=0), len(removed_edges))
        ranks = np.array([normalized_rank(np.asarray([S1Model.get_edge_prob(embedding, graph_info.average_degree, i, j) for (i, j) in all_non_edges_updated]), len(removed_edges))
                            for embedding in embeddings])
        bigue_ranks_average = np.average(ranks, axis=0)
        bigue_auc = [(get_auc(original_graph, get_matrix_prob(embedding, graph_info.average_degree))) for embedding in embeddings]
        mercator_auc = get_auc(original_graph, get_matrix_prob(mercator_embedding, graph_info.average_degree))

        bigue_auc_stream.write(' '.join(list(map(str, bigue_auc))) + ' ')
        bigue_rank_average_stream.write(' '.join(list(map(str, bigue_ranks_average))) + ' ')
        bigue_sample_average_stream.write(' '.join(list(map(str, bigue_prob_average))) + ' ')
        mercator_rank_stream.write(' '.join(list(map(str, mercator_ranks))) + ' ')
        mercator_auc_stream.write(str(mercator_auc) + ' ')
        if groundtruth_parameters is not None:
            gt_rank_stream.write(' '.join(list(map(str, gt_ranks))) + ' ')
