import os
import re

import numpy as np
import basegraph
from pybigue.embedding_info import EmbeddingParameters
from pybigue.models import S1Model, angular_distance
from pybigue.utils import align_theta

from results.paths import project_dir, write_dataclass



def embed_with_mercator(path_manager, fixed_vertices, score_function, repetitions=10, verbose=True, **kwargs):
    best_embedding = EmbeddingParameters()
    best_loglikelihood = -np.inf
    for i in range(repetitions):
        if verbose:
            print(f"Embedding with Mercator {i+1}/{repetitions}", end="\r")
        _embed_with_mercator(path_manager, **kwargs)
        embedding = _read_mercator_embedding(path_manager, fixed_vertices)
        val = score_function(embedding)
        if val > best_loglikelihood:
            best_loglikelihood = val
            best_embedding = embedding
    if verbose:
        print()
    write_dataclass(best_embedding, path_manager.mercator_embedding_path)
    return best_embedding


def _embed_with_mercator(path_manager, beta=None, refinements=10):
    mercator = os.path.join(project_dir, "mercator", "mercator")
    os.system(
        f"{mercator} {'' if beta is None else f'-b {beta}'} {path_manager.graph_path}"
    )
    for _ in range(refinements):
        os.system(
            f"{mercator} -r {path_manager.mercator_path} {path_manager.graph_path}"
        )


def _read_mercator_embedding(path_manager, fixed_vertices):
    embedding_params = np.loadtxt(path_manager.mercator_path)
    n = len(embedding_params[:, 1])
    if np.any(np.arange(n) != embedding_params[:, 0]):
        raise ValueError(f"Vertices {np.argwhere(np.arange(n) != embedding_params[:, 0]).tolist()} "
                         "are missing in Mercator embedding")

    theta = embedding_params[:, 2] - np.pi  # mercator returns angles in [0, 2pi]
    kappa = embedding_params[:, 1]

    beta = None
    with open(path_manager.mercator_path, "r") as file_stream:
        for line in file_stream.readlines():
            if (result := re.match(r"#\s+- beta:\s+([0-9]+\.[0-9]*)", line)):
                beta = np.double(result[1])
    if beta is None:
        raise ValueError("Couldn't find Mercator's inferred beta.")

    return EmbeddingParameters(theta=align_theta(theta, *fixed_vertices),
                               kappa=kappa.tolist(),
                               beta=beta)


def filter_graph(graph, force_connected=True, min_degree=2, verbose=True):
    n = graph.get_size()
    if force_connected:
        components = basegraph.metrics.find_weakly_connected_components(graph)
        largest_component = np.argmax(list(map(len, components)))
        kept_vertices = set(components[largest_component])
    else:
        kept_vertices = set(range(graph.get_size()))

    if len(kept_vertices) < n and verbose:
        print(f"Filtering reduced graph size from {n} to {len(kept_vertices)}.")

    filtered_graph = graph

    def get_vertices_to_remove():
        return [
            i for i in kept_vertices if 0 < graph.get_degree(i) < min_degree
        ]

    vertices_to_remove = get_vertices_to_remove()
    while len(kept_vertices) > 1 and len(vertices_to_remove) > 0:
        for vertex in vertices_to_remove:
            filtered_graph.remove_vertex_from_edgelist(vertex)
            kept_vertices.remove(vertex)
        vertices_to_remove = get_vertices_to_remove()

    if len(kept_vertices) < min_degree:
        raise ValueError("Filtered graph empty.")

    return basegraph.core.algorithms.find_subgraph_with_remap(
        filtered_graph, kept_vertices)


def remap_embedding(groundtruth_parameters: EmbeddingParameters, new_mapping, graph_info):
    return EmbeddingParameters(
        theta=align_theta(
            remap_parameter(groundtruth_parameters.theta, new_mapping),
            *graph_info.fixed_vertices
        ),
        kappa=remap_parameter(groundtruth_parameters.kappa, new_mapping).tolist(),
        beta=groundtruth_parameters.beta)


def remap_parameter(parameter, new_map):
    remapped_parameter = np.zeros(len(new_map))
    for old, new in new_map.items():
        remapped_parameter[new] = parameter[old]
    return remapped_parameter


def hyperbolic_distance(r1, r2, theta1, theta2):
    if theta1 == theta2:
        return np.abs(r1 - r2)

    return np.arccosh(
        np.cosh(r1) * np.cosh(r2) -
        np.sinh(r1) * np.sinh(r2) * np.cos(theta1 - theta2))
