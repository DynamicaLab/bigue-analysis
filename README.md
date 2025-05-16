# Numerical analysis of BIGUE

This repository contains the code used to generate the Figures presented in the paper introducing BIGUE. The code heavily relies on [pybigue].

The repository is shared for reference and is minimally documented. Only the commands necessary to generate the figures are given.

If you use this algorithm, please cite:

[Simon Lizotte](https://siliz4.github.io), [Jean-Gabriel Young](https://jgyoung.ca) and [Antoine Allard](https://antoineallard.github.io)
"Symmetry-driven embedding of networks in hyperbolic space". _Commun. Phys._ __8__, 199 (2025).
https://doi.org/10.1038/s42005-025-02122-0

## Installation

First download the code and its submodules using git
```sh
git clone https://github.com/DynamicaLab/bigue-analysis.git
```
Install [pybigue] and the requirements of this code
```
pip install pybigue
pip install -r pybigue-analysis/requirements.txt
```
Install [Basegraph](https://github.com/BaseGraph/BaseGraph.git) and its [metrics](https://github.com/BaseGraph/BaseGraphMetrics) and geometry extensions as Python modules (extensions were copied in BaseGraph to ease installation): execute in the `basegraph_libs` directory
```
pip install ./BaseGraph
pip install ./BaseGraphMetrics
pip install ./BaseGraphGeometry
```

Compile Mercator embedding's program as `mercator` with (example using g++ in the `mercator` directory)
```{sh}
g++ -O3 src/embeddingS1_unix.cpp -Iinclude -o mercator
```

## Reproducing the paper figures

The `.raw_data` directory contains the results of the simulations used in the paper.
This means that the Figures can be generated without running the simulations.
The `.data` directory contains the edge lists and the ground truth embeddings used in the paper:

- `100v.txt`: Edge list of the 100 vertices synthetic graph used in Supplementary Figure 1.
- `100v_embedding.json`: Parameters used to generate the 100 vertices synthetic graph.
- `30v.txt`: Edge list of the 30 vertices synthetic graph used in Figure 1.
- `30v_embedding.json`: Parameters used to generate the 30 vertices synthetic graph.
- `conflicting.txt`: Edge list of the synthetic graph with a conflicting embedding (Fig. 6c).
- `conflicting_embedding1.json`: The first embedding used to generate the conflicting synthetic graph.
- `conflicting_embedding2.json`: The second embedding used to generate the conflicting synthetic graph.
- `karate.txt`: Edge list of Zachary's karate club (Zachary, Fig. 7).
- `macaque_neural.txt`: Edge list of the cortical connectivity of the macaque (Macaque, Fig. 7).
- `montreal.txt`: Edge list of the street gangs in Montreal (Gangs, Fig. 7).
- `dutch_criticism.txt`: Edge list of the dutch literary critics (Critics, Fig. 7).
- `zebras.txt`: Edge list zebra social interactions (Zebras, Fig. 7).
- `november17.txt`: Edge list of the connections in greek terrorist group (Terrorism, Fig. 7).
- `kangaroo.txt`: Edge list of the kangaroo dominance relationships (Kangaroo, Fig. 7).
- `new_guinea_tribes.txt`: Edge list of the New Guinea tribes friendships (Tribes, Fig. 7).

The algorithm settings used in the simulations are in the corresponding file in the `config` directory.

The scripts must be ran from the root directory for the style of `matplotlibrc` to take effect.
Figures are saved in the `figures/` directory.

Figure 1:
```sh
python results/hyperbolic_errorbars.py -i groundtruth 30v_long
```

Figure 2:
```sh
python results/vertex_likelihood.py config/30v_long.yaml
```

Figure 3: Combination of two figures
```sh
python results/fig_autocovariance.py
python results/fig_compare_algs.py
```

Figure 5: Combination of multiple figures
```sh
python results/errorbars.py config/30v_long.yaml -i groundtruth
python results/compare_properties.py config/30v_long.yaml -i groundtruth
python results/link_prediction_hist.py config/30v_long.yaml
```

Figure 6:
```sh
python results/fig_marginals.py
```

Figure 7:
```sh
python results/fig_empirical_properties.py
```

Supplementary Figure 1
```sh
python results/errorbars.py config/100v.yaml
python results/autocovariance.py config/100v.yaml
```

Supplementary Figure 2
```sh
python results/fig_distance_approx.py
python results/vertex_approx_likelihood.py config/30v_long.yaml
```

Supplementary Figure 3: Combination of two figures
```sh
python results/link_prediction_hist.py config/30v_long.yaml
python results/auc.py config/30v_long.yaml
```

## Reproducing simulations

This section details the commands used to run the simulations.
Each script can be executed in parallel (e.g. using GNU parallel). To generate the necessary samples for each graph, run
```sh
for dataset in conflicting.yaml dutch_criticism.yaml kangaroo.yaml karate.yaml macaque_neural.yaml montreal.yaml new_guinea_tribes.yaml november17.yaml zebras.yaml; do
    python sample.py config/$dataset -i random
done
synthetic_datasets=(
for dataset in 30v_rw.yaml 30v_hmc.yaml 30v.yaml 30v_long.yaml 30v_rw_long.yaml); do
    python sample.py config/$dataset -i random groundtruth --groundtruth ./.data/30v.txt ./.data/30v_embedding.json
done
python sample.py config/$dataset -i random groundtruth --groundtruth ./.data/100v.txt ./.data/100v_embedding.json
```
Figure 5h requires running the script (each iteration can also be parallelized)
```sh
for i in 0..20; do
    python link_prediction.py $i -N 20
done
```

[pybigue]: https://github.com/DynamicaLab/bigue
