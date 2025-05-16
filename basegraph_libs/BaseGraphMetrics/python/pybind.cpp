#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "BaseGraph/algorithms/paths.hpp"
#include "BaseGraph/directed_graph.hpp"
#include "BaseGraph/extensions/metrics/directed.h"
#include "BaseGraph/extensions/metrics/general.h"
#include "BaseGraph/extensions/metrics/undirected.h"
#include "BaseGraph/undirected_graph.hpp"

namespace py = pybind11;
using namespace BaseGraph;


// From https://github.com/pybind/pybind11/issues/1042
template <typename Sequence>
inline py::array_t<typename Sequence::value_type> as_pyarray(Sequence &&seq) {
    // Move entire object to heap (Ensure is moveable!). Memory handled via
    // Python capsule
    Sequence *seq_ptr = new Sequence(std::move(seq));
    auto capsule = py::capsule(
        seq_ptr, [](void *p) { delete reinterpret_cast<Sequence *>(p); });
    return py::array(seq_ptr->size(), // shape of array
                     seq_ptr->data(), // c-style contiguous strides for Sequence
                     capsule          // numpy array references this parent
    );
}

template <template <class...> class Graph, typename EdgeLabel>
void declareGeneralMetrics(py::module& m) {
    m.def("get_closeness_centralities", [&](const Graph<EdgeLabel> &graph) {
        return as_pyarray(metrics::getClosenessCentralities(graph));
    });
    m.def("get_harmonic_centralities", [&](const Graph<EdgeLabel> &graph) {
        return as_pyarray(metrics::getHarmonicCentralities(graph));
    });
    m.def("get_betweenness_centralities",
          [&](const Graph<EdgeLabel> &graph, bool normalize) {
              return as_pyarray(
                  metrics::getBetweennessCentralities(graph, normalize));
          });
    m.def("get_shortest_path_lengths_from_vertex",
          [&](const Graph<EdgeLabel> &graph, VertexIndex vertex) {
              return as_pyarray(
                  metrics::getShortestPathLengthsFromVertex(graph, vertex));
          });
/**/m.def("get_diameters", [](const Graph<EdgeLabel> &graph) {
              return as_pyarray(metrics::getDiameters(graph));
          });
    m.def("get_shortest_path_averages", [](const Graph<EdgeLabel> &graph) {
        return as_pyarray(metrics::getShortestPathAverages(graph));
    });
    m.def("get_shortest_path_harmonic_averages",
          py::overload_cast<const Graph<EdgeLabel> &>(
              &metrics::getShortestPathHarmonicAverages<Graph, EdgeLabel>));

/**/m.def("get_shortest_paths_distribution",
          py::overload_cast<const Graph<EdgeLabel> &>(
              &metrics::getShortestPathsDistribution<Graph, EdgeLabel>));
/**/m.def("find_weakly_connected_components",
               py::overload_cast<const Graph<EdgeLabel> &>(
                   &metrics::findWeaklyConnectedComponents<Graph, EdgeLabel>));
}

template<typename EdgeLabel>
void declareGeneralMetricsBothOrientations(py::module& m) {
    declareGeneralMetrics<LabeledDirectedGraph, EdgeLabel>(m);
    declareGeneralMetrics<LabeledUndirectedGraph, EdgeLabel>(m);
}

template <typename EdgeLabel>
void declareDirectedMetrics(py::module& m) {
    m.def("get_density", &metrics::getDensity<EdgeLabel>);
/**/m.def("find_all_directed_triangles",
          py::overload_cast<const LabeledDirectedGraph<EdgeLabel> &>(
              &metrics::findAllDirectedTriangles<EdgeLabel>));
/**/m.def("get_triangle_spectrum", &metrics::getTriangleSpectrum<EdgeLabel>);
    m.def("get_undirected_local_clustering_coefficients",
          py::overload_cast<const LabeledDirectedGraph<EdgeLabel> &>(
              &metrics::getUndirectedLocalClusteringCoefficients<EdgeLabel>));
    m.def("get_undirected_global_clustering_coefficient",
          py::overload_cast<const LabeledDirectedGraph<EdgeLabel> &>(
              &metrics::getUndirectedGlobalClusteringCoefficient<EdgeLabel>));

    m.def("get_reciprocity", &metrics::getReciprocity<EdgeLabel>);
/**/m.def("get_reciprocal_degrees", &metrics::getReciprocalDegrees<EdgeLabel>);
/**/m.def("get_jaccard_reciprocities",
          py::overload_cast<const LabeledDirectedGraph<EdgeLabel> &>(
              &metrics::getJaccardReciprocities<EdgeLabel>));
/**/m.def("get_reciprocity_ratios",
          py::overload_cast<const LabeledDirectedGraph<EdgeLabel> &>(
              &metrics::getReciprocityRatios<EdgeLabel>));
/**/m.def("get_out_degree_histogram", &metrics::getOutDegreeHistogram<EdgeLabel>);
/**/m.def("get_in_degree_histogram",
          py::overload_cast<const LabeledDirectedGraph<EdgeLabel> &>(
              &metrics::getInDegreeHistogram<EdgeLabel>));
}

template <typename EdgeLabel>
void declareUndirectedMetrics(py::module& m) {
    m.def("get_degree_correlation", py::overload_cast<const LabeledUndirectedGraph<EdgeLabel> &>(
                                        &metrics::getDegreeCorrelation<EdgeLabel>));
    m.def("find_all_triangles", &metrics::findAllTriangles<EdgeLabel>);
    m.def("count_triangles_around_vertex", py::overload_cast<const LabeledUndirectedGraph<EdgeLabel>&, VertexIndex>(
          &metrics::countTrianglesAroundVertex<EdgeLabel>));
    m.def("count_triangles", &metrics::countTriangles<EdgeLabel>);

    m.def("get_local_clustering_coefficients",
          py::overload_cast<const LabeledUndirectedGraph<EdgeLabel> &>(
              &metrics::getLocalClusteringCoefficients<EdgeLabel>));
    m.def("get_global_clustering_coefficient",
          py::overload_cast<const LabeledUndirectedGraph<EdgeLabel> &>(
              &metrics::getGlobalClusteringCoefficient<EdgeLabel>));
    m.def("get_clustering_spectrum", &metrics::getClusteringSpectrum<EdgeLabel>);
    m.def("get_redundancy", &metrics::getRedundancy<EdgeLabel>);

    m.def("get_kshells_and_onion_layers", &metrics::getKShellsAndOnionLayers<EdgeLabel>);
    m.def("get_kshells", &metrics::getKShells<EdgeLabel>);
    m.def("get_onion_layers", &metrics::getOnionLayers<EdgeLabel>);
    m.def("get_onion_spectrum", py::overload_cast<const LabeledUndirectedGraph<EdgeLabel> &>(
                                    &metrics::getOnionSpectrum<EdgeLabel>));
/**/m.def("get_kcore", py::overload_cast<const LabeledUndirectedGraph<EdgeLabel> &, size_t>(
                           &metrics::getKCore<EdgeLabel>));
    m.def("get_neighbour_degrees", &metrics::getNeighbourDegrees<EdgeLabel>);
    m.def("get_neighbour_degree_spectrum",
          &metrics::getNeighbourDegreeSpectrum<EdgeLabel>);

    m.def("get_modularity", &metrics::getModularity<EdgeLabel>);
}

// Metrics that aren't validated with Networkx are tagged with /**/
PYBIND11_MODULE(_metrics, m) {
    // Required Python import the module to work
    py::module_::import("basegraph");

    // General metrics
    declareGeneralMetricsBothOrientations<NoLabel>(m);
    declareGeneralMetricsBothOrientations<std::string>(m);

    // Undirected metrics
    declareUndirectedMetrics<NoLabel>(m);
    declareUndirectedMetrics<std::string>(m);

    // Directed metrics
    declareDirectedMetrics<NoLabel>(m);
    declareDirectedMetrics<std::string>(m);
}
