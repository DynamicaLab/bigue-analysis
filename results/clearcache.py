import os
from argparse import ArgumentParser
from pathlib import Path
from shutil import rmtree

from paths import ResultPaths


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config_file", help="Config file path")
    parser.add_argument("-i", "--init", nargs="+",
                        default=["random", "groundtruth", "mercator"],
                        help="Initialization", choices=["random", "groundtruth", "mercator"])
    args = parser.parse_args()

    config_name = os.path.splitext(Path(args.config_file).name)[0]

    path_manager = ResultPaths(config_name)
    for init in args.init:
        path_manager.sample_prefix = init
        if os.path.isdir(path_manager.cache_dir_path):
            print("Deleting cache")
            rmtree(path_manager.cache_dir_path)
        else:
            print("Cache empty")
