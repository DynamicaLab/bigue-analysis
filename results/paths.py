import json
import os
import pathlib
from dataclasses import dataclass, asdict

import basegraph
import numpy as np


project_dir = str(pathlib.Path(__file__).parent.parent.resolve())
@dataclass
class ResultPaths:
    config_name: str
    output_dir_path: str = os.path.join(project_dir, ".raw_data")
    sample_dir: str = "sample"
    sample_prefix: str = ""
    cache_dir : str = "cache"

    graph_information_file: str = "graph_info.json"
    groundtruth_embedding_file: str = "groundtruth_embedding.json"
    graph_file: str = "graph.txt"
    mercator_embedding_file: str = "mercator_embedding.json"

    @property
    def config_output_path(self):
        return os.path.join(self.output_dir_path, self.config_name)

    @property
    def sample_dir_path(self):
        return os.path.join(self.config_output_path, self.sample_prefix + self.sample_dir)

    @property
    def cache_dir_path(self):
        return os.path.join(self.config_output_path, self.sample_prefix + self.cache_dir)

    @property
    def groundtruth_embedding_path(self):
        return os.path.join(self.config_output_path, self.groundtruth_embedding_file)

    @property
    def graph_path(self):
        return os.path.join(self.config_output_path, self.graph_file)

    @property
    def graph_information_path(self):
        return os.path.join(self.config_output_path, self.graph_information_file)


    @property
    def mercator_embedding_path(self):
        return os.path.join(
            self.config_output_path, self.mercator_embedding_file)

    @property
    def mercator_path(self):
        return os.path.join(
            self.config_output_path,
            os.path.splitext(self.graph_file)[0] + ".inf_coord")


def write_graph(graph: basegraph.core.UndirectedGraph, filename: str):
    basegraph.core.io.write_text_edgelist(graph, filename)


def read_graph(filename: str):
    return basegraph.core.io.load_undirected_text_edgelist_indexed(filename)[0]


def write_dataclass(object, filename: str):
    with open(filename, 'w') as file_stream:
        json.dump(asdict(object), file_stream, cls=NumpyJSONEncoder)


def read_dataclass(Dataclass_type, filename: str):
    with open(filename, 'r') as file_stream:
        return Dataclass_type(**{key: np.asarray(val) if isinstance(val, list) else val for (key, val) in json.load(file_stream).items()})


class NumpyJSONEncoder(json.JSONEncoder):

    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)
