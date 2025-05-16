import os
import warnings
from argparse import ArgumentParser
from itertools import chain

import basegraph
import numpy as np

from pybigue.models import S1Model
from pybigue.utils import gen_cauchy_lpdf, gen_normal_lpdf, sample_truncated_pareto
from pybigue.sampling import (
        read_sample, sample_chain, run_parallel_chains, sample_hmc,
        sample_truncated_normal, sample_uniform)
from pybigue.embedding_info import GraphInformation, EmbeddingParameters, replace_known_parameters
from pybigue.kernels.transforms import get_global_sampling_kernel
from pybigue.utils import align_theta

from config import parse_config
from util import embed_with_mercator, remap_embedding, filter_graph
from results.paths import read_dataclass, read_graph, write_dataclass, write_graph
from results.modelingtools import generate_graph
import results.paths as paths


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config_file", help="Config file path")
    parser.add_argument("-c", "--continue", dest="continue_",
                        help="Continue from last simulation", action="store_true")
    parser.add_argument("-i", "--init", nargs="+", default=["random", "groundtruth", "mercator"],
                        help="Initialization", choices=["random", "groundtruth", "mercator"])
    parser.add_argument("--groundtruth", nargs=2, help="Graph file and ground truth parameters (respectively). Bypasses the synthetic generation.")
    args = parser.parse_args()

    config = parse_config(args.config_file)
    alg_parameters = config["algorithm"]

    path_manager = paths.ResultPaths(config["name"])
    if not os.path.isdir(path_manager.config_output_path):
        os.makedirs(path_manager.config_output_path)

    if seed := config.get("seed"):
        np.random.seed(seed)

    hyperparameters = config["hyperparameters"]
    if graph := config.get("graph"):
        print("Reading existing graph")
        if alg_parameters.filter_graph:
            graph, _ = filter_graph(graph)
        graph_info = GraphInformation.from_degrees(graph.get_degrees())
        groundtruth_parameters = None
    elif args.groundtruth:
        graph = read_graph(args.groundtruth[0])
        groundtruth_parameters = read_dataclass(EmbeddingParameters, args.groundtruth[1])
        graph_info = GraphInformation.from_degrees(graph.get_degrees())
        write_dataclass(groundtruth_parameters,
                        path_manager.groundtruth_embedding_path)
    else:
        # Graph generation
        gen_parameters = config["generation"]
        groundtruth_parameters = EmbeddingParameters(
            theta=sample_uniform(-np.pi, np.pi, gen_parameters.n),
            kappa=sample_truncated_pareto(gen_parameters.kappa_min,
                                          gen_parameters.kappa_max,
                                          gen_parameters.gamma,
                                          size=gen_parameters.n),
            beta=gen_parameters.beta)

        graph = generate_graph(groundtruth_parameters, np.average(groundtruth_parameters.kappa))
        if alg_parameters.filter_graph:
            graph, new_mapping = filter_graph(graph)
            graph_info = GraphInformation.from_degrees(graph.get_degrees())
            groundtruth_parameters = remap_embedding(groundtruth_parameters, new_mapping, graph_info)
        else:
            graph_info = GraphInformation.from_degrees(graph.get_degrees())
        write_dataclass(groundtruth_parameters,
                        path_manager.groundtruth_embedding_path)

    vertex_number = graph.get_size()
    write_dataclass(graph_info, path_manager.graph_information_path)
    write_graph(graph, path_manager.graph_path)

    # Embed
    inferred_parameters = set(chain.from_iterable([params.get("for")
                                                   for params in alg_parameters.kernels.values()
                                                   if params.get("for") is not None]))
    if inferred_parameters == []:
        raise ValueError("No parameter is inferred.")
    if groundtruth_parameters is not None:
        known_parameters = EmbeddingParameters(**{
                    param: None if param in inferred_parameters else groundtruth_parameters[param]
                          for param in EmbeddingParameters.names()
            })
    else:
        if not_inferred := (set(EmbeddingParameters.names()) - inferred_parameters):
            raise ValueError(f"Parameters {not_inferred} are not inferred while all parameters must be inferred.")
        known_parameters = None


    # kappa_logprior = gen_pareto_lpdf(hyperparameters.gamma, hyperparameters.kappa_min) if "kappa" in inferred_parameters else lambda *_: 0
    kappa_logprior = gen_cauchy_lpdf(0, hyperparameters.gamma) if "kappa" in inferred_parameters else lambda *_: 0
    beta_logprior = gen_normal_lpdf(hyperparameters.beta_average, hyperparameters.beta_std) if "beta" in inferred_parameters else lambda *_: 0
    adjacency = np.array(graph.get_adjacency_matrix(False))

    def logposterior(embedding):
        return S1Model.loglikelihood(adjacency, graph_info.average_degree, embedding.theta, embedding.kappa, embedding.beta)\
                + np.sum(kappa_logprior(embedding.kappa)) + beta_logprior(embedding.beta)

    if os.path.isfile(path_manager.mercator_embedding_path):
        print("Skipping Mercator embedding which was already computed.")
        mercator_embedding = read_dataclass(EmbeddingParameters, path_manager.mercator_embedding_path)
    else:
        if len(basegraph.metrics.find_weakly_connected_components(graph))>1:
            print("Not running Mercator: graph is not connected.")
            mercator_embedding = None
        else:
            mercator_embedding = embed_with_mercator(path_manager, graph_info.fixed_vertices,
                                                                 lambda x: S1Model.loglikelihood(adjacency, graph_info.average_degree, **x.as_dict()), verbose=False)

    for init_strategy in args.init:
        path_manager.sample_prefix = init_strategy
        sampling_directory = path_manager.sample_dir_path
        if os.path.isdir(sampling_directory):
            if os.listdir(sampling_directory) and not args.continue_:
                print("Previous inference results will be overwritten.")
        else:
            os.makedirs(sampling_directory)

        if args.continue_:
            print(f"Further sampling with {init_strategy} initialization.")
            previous_samples = read_sample(sampling_directory)
            if "hmc" in alg_parameters.kernels:
                raise NotImplementedError("Not supported with hmc.")
            if len(previous_samples) == 0:
                print("No previous sample.")
                exit()

            def init(chain):
                return replace_known_parameters(previous_samples[str(chain)].end(), known_parameters)
        else:
            print(f"Sampling with {init_strategy} initialization.")
            match init_strategy:
                case "random":
                    def init(_):
                        embedding = replace_known_parameters(EmbeddingParameters(
                                        theta=sample_uniform(-np.pi, np.pi, vertex_number),
                                        kappa=np.array([max(1e-10, deg) for deg in graph.get_degrees()]),
                                        beta=sample_truncated_normal(hyperparameters.beta_average, hyperparameters.beta_std, size=1, lower_bound=1)),
                                 known_parameters
                        )
                        embedding.theta = align_theta(embedding.theta, *graph_info.fixed_vertices)
                        return embedding

                case "groundtruth":
                    if groundtruth_parameters is None:
                        warnings.warn("Cannot use ground truth initialization since it's unknown.")
                        continue
                    def init(_):
                        embedding = groundtruth_parameters
                        embedding.theta = align_theta(embedding.theta, *graph_info.fixed_vertices)
                        return embedding

                case "mercator":
                    if mercator_embedding is None:
                        warnings.warn("Cannot use Mercator initialization.")
                        continue
                    def init(_):
                        return replace_known_parameters(mercator_embedding, groundtruth_parameters)

                case _:
                    raise ValueError("Unknown initialization scheme.")

        kernel_args = (alg_parameters.kernels, init(0), known_parameters, graph, graph_info, hyperparameters, logposterior)
        kernel = get_global_sampling_kernel(*kernel_args)

        if list(alg_parameters.kernels.keys()) == ["hmc"]:
            sample_mcmc = lambda chain_id, *_: sample_hmc(
                    kernel,
                    init,
                    sampling_directory,
                    sample_size=alg_parameters.sample_size,
                    thin=alg_parameters.thin,
                    chain_id=chain_id,
                    merge_samples=args.continue_,
                )
        else:
            sample_mcmc = lambda chain_id, log_progress: sample_chain(
                    kernel,
                    init,
                    sampling_directory,
                    sample_size=alg_parameters.sample_size,
                    warmup=0 if args.continue_ else alg_parameters.warmup,
                    thin=alg_parameters.thin,
                    chain_id=chain_id,
                    log_progress=log_progress,
                    merge_samples=args.continue_,
                )

        run_parallel_chains(sample_mcmc, chain_number=alg_parameters.chain_number)
