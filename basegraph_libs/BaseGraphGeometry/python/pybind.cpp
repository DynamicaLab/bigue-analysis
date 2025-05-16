#include <cmath>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "BaseGraph/undirected_graph.hpp"
#include "BaseGraph/extensions/geometry/metrics.hpp"

namespace py = pybind11;
using namespace BaseGraph;


template <typename EdgeLabel>
void declareMetrics(py::module& m) {
    m.def("get_greedy_stability", [&](
            const LabeledUndirectedGraph<EdgeLabel> &graph,
            const std::vector<std::vector<double>> &positions) {
        return geometry::getGreedyStability(graph, positions, geometry::distanceH2);
    });
    m.def("get_greedy_routing_scores", [&](
                const LabeledUndirectedGraph<EdgeLabel> &graph,
                const std::vector<std::vector<double>> &positions) {
            return geometry::getGreedyRoutingScores(graph, positions, geometry::distanceH2);
        });
    m.def("get_hierarchy_levels", [&](
                const LabeledUndirectedGraph<EdgeLabel> &graph,
                const std::vector<double> &angles,
                const std::vector<double> &radii) {
            return geometry::getHierarchyLevel(graph, angles, radii);
        });
}

PYBIND11_MODULE(_geometry, m) {
    // Required Python import the module to work
    py::module_::import("basegraph");

    declareMetrics<NoLabel>(m);
    declareMetrics<std::string>(m);
}
