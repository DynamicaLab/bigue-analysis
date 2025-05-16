#ifndef BASE_GRAPH_GENERAL_GRAPH_METRICS_H
#define BASE_GRAPH_GENERAL_GRAPH_METRICS_H

#include <list>
#include <queue>
#include <unordered_map>
#include <vector>

#include "BaseGraph/algorithms/paths.hpp"
#include "BaseGraph/directed_graph.hpp"
#include "BaseGraph/types.h"
#include "BaseGraph/undirected_graph.hpp"

namespace BaseGraph {
namespace metrics {

typedef std::list<VertexIndex> Component;

template <template <class...> class Graph, typename EdgeLabel>
std::vector<size_t>
getShortestPathLengthsFromVertex(const Graph<EdgeLabel> &graph,
                                 VertexIndex source) {
    return algorithms::findVertexPredecessors(graph, source).first;
}

template <template <class...> class Graph, typename EdgeLabel>
std::vector<size_t> getDiameters(const Graph<EdgeLabel> &graph) {
    size_t verticesNumber = graph.getSize();
    std::vector<size_t> diameters(verticesNumber);

    std::vector<size_t> shortestPathLengths;
    size_t largestDistance;
    for (const VertexIndex &i : graph) {
        shortestPathLengths = getShortestPathLengthsFromVertex(graph, i);
        largestDistance = shortestPathLengths[0];

        for (const VertexIndex &j : graph)
            if (shortestPathLengths[j] > largestDistance &&
                shortestPathLengths[j] != algorithms::BASEGRAPH_VERTEX_MAX)
                largestDistance = shortestPathLengths[j];

        if (largestDistance == algorithms::BASEGRAPH_VERTEX_MAX)
            diameters[i] = 0;
        else
            diameters[i] = largestDistance;
    }
    return diameters;
}

template <template <class...> class Graph, typename EdgeLabel>
double getShortestPathAverage(const Graph<EdgeLabel> &graph,
                                     VertexIndex vertex) {
    std::vector<size_t> shortestPathLengths =
        getShortestPathLengthsFromVertex(graph, vertex);

    size_t sum = 0;
    size_t componentSize = 1;

    for (VertexIndex &vertex : graph)
        if (shortestPathLengths[vertex] != 0 &&
            shortestPathLengths[vertex] != algorithms::BASEGRAPH_VERTEX_MAX) {
            sum += shortestPathLengths[vertex];
            componentSize++;
        }

    return componentSize > 1 ? (double)sum / (componentSize - 1) : 0;
}

template <template <class...> class Graph, typename EdgeLabel>
std::vector<double> getShortestPathAverages(const Graph<EdgeLabel> &graph) {
    std::vector<double> averageShortestPaths(graph.getSize(), 0);

    for (VertexIndex vertex : graph)
        averageShortestPaths[vertex] = getShortestPathAverage(graph, vertex);

    return averageShortestPaths;
}

template <template <class...> class Graph, typename EdgeLabel>
double getClosenessCentrality(const Graph<EdgeLabel> &graph,
                              VertexIndex vertex) {
    std::vector<size_t> shortestPathLengths =
        getShortestPathLengthsFromVertex(graph, vertex);
    size_t componentSize = 0;
    unsigned long long int sum = 0;

    for (VertexIndex &vertex : graph) {
        if (shortestPathLengths[vertex] != algorithms::BASEGRAPH_VERTEX_MAX) {
            componentSize += 1;
            sum += shortestPathLengths[vertex];
        }
    }
    return sum > 0 ? ((double)componentSize - 1) / sum : 0;
}

template <template <class...> class Graph, typename EdgeLabel>
std::vector<double> getClosenessCentralities(const Graph<EdgeLabel> &graph) {
    std::vector<double> closenessCentralities(graph.getSize(), 0);

    for (VertexIndex vertex : graph)
        closenessCentralities[vertex] = getClosenessCentrality(graph, vertex);

    return closenessCentralities;
}

template <template <class...> class Graph, typename EdgeLabel>
double getHarmonicCentrality(const Graph<EdgeLabel> &graph,
                             VertexIndex vertex) {
    std::vector<size_t> shortestPathLengths =
        getShortestPathLengthsFromVertex(graph, vertex);

    double harmonicSum = 0;
    for (VertexIndex &vertex : graph)
        if (shortestPathLengths[vertex] != 0 &&
            shortestPathLengths[vertex] != algorithms::BASEGRAPH_VERTEX_MAX)
            harmonicSum += 1.0 / shortestPathLengths[vertex];

    return harmonicSum;
}

template <template <class...> class Graph, typename EdgeLabel>
std::vector<double> getHarmonicCentralities(const Graph<EdgeLabel> &graph) {
    std::vector<double> harmonicCentralities(graph.getSize(), 0);

    for (VertexIndex vertex : graph)
        harmonicCentralities[vertex] = getHarmonicCentrality(graph, vertex);

    return harmonicCentralities;
}

template <template <class...> class Graph, typename EdgeLabel>
std::list<Component>
findWeaklyConnectedComponents(const Graph<EdgeLabel> &graph) {
    size_t verticesNumber = graph.getSize();
    if (verticesNumber == 0)
        throw std::logic_error("There are no vertices.");

    std::list<Component> connectedComponents;
    Component currentComponent;
    VertexIndex currentVertex, startVertex;

    std::queue<VertexIndex> verticesToProcess;
    std::vector<bool> processedVertices;
    bool allVerticesProcessed = false;
    processedVertices.resize(verticesNumber, false);

    std::list<VertexIndex>::const_iterator vertexNeighbour;

    while (!allVerticesProcessed) {
        allVerticesProcessed = true;
        for (VertexIndex i = 0; i < verticesNumber && allVerticesProcessed;
             ++i) {
            if (!processedVertices[i]) {
                allVerticesProcessed = false;
                startVertex = i;
            }
        }

        if (!allVerticesProcessed) {
            currentComponent.clear();
            verticesToProcess.push(startVertex);
            processedVertices[startVertex] = true;

            while (!verticesToProcess.empty()) {
                currentVertex = verticesToProcess.front();

                for (const VertexIndex &vertexNeighbour :
                     graph.getOutNeighbours(currentVertex)) {
                    if (!processedVertices[vertexNeighbour]) {
                        verticesToProcess.push(vertexNeighbour);
                        processedVertices[vertexNeighbour] = true;
                    }
                }
                currentComponent.push_back(currentVertex);
                verticesToProcess.pop();
            }
            connectedComponents.push_back(currentComponent);
        }
    }
    return connectedComponents;
}

template <template <class...> class Graph, typename EdgeLabel>
std::vector<std::unordered_map<size_t, double>>
getShortestPathsDistribution(const Graph<EdgeLabel> &graph) {
    auto connectedComponents = findWeaklyConnectedComponents(graph);

    std::vector<size_t> shortestPathLengths;
    std::vector<std::unordered_map<size_t, double>> shortestPathDistribution(
        connectedComponents.size());
    size_t componentIndex = 0;

    for (auto component : connectedComponents) {
        auto &currentDistribution = shortestPathDistribution[componentIndex];

        if (component.size() > 1) {
            for (const VertexIndex &vertex : component) {
                shortestPathLengths =
                    getShortestPathLengthsFromVertex(graph, vertex);

                for (const size_t &pathLength : shortestPathLengths) {
                    if (pathLength != 0 &&
                        pathLength != algorithms::BASEGRAPH_VERTEX_MAX) {
                        if (currentDistribution.find(pathLength) ==
                            currentDistribution.end())
                            currentDistribution[pathLength] = 1;
                        else
                            currentDistribution[pathLength]++;
                    }
                }
            }
            for (auto &element : currentDistribution)
                element.second /= component.size();
        }

        componentIndex++;
    }
    return shortestPathDistribution;
}

template <template <class...> class Graph, typename EdgeLabel>
double getShortestPathHarmonicAverage(const Graph<EdgeLabel> &graph,
                                             VertexIndex vertex) {
    std::vector<size_t> shortestPathLengths =
        getShortestPathLengthsFromVertex(graph, vertex);
    size_t componentSize = 0;

    double sumOfInverse = 0;
    for (VertexIndex &vertex : graph) {
        if (shortestPathLengths[vertex] != 0 &&
            shortestPathLengths[vertex] != algorithms::BASEGRAPH_VERTEX_MAX) {
            componentSize += 1;
            sumOfInverse += 1.0 / shortestPathLengths[vertex];
        }
    }
    return componentSize > 0 ? sumOfInverse / componentSize : 0;
}

template <template <class...> class Graph, typename EdgeLabel>
std::vector<double>
getShortestPathHarmonicAverages(const Graph<EdgeLabel> &graph) {
    std::vector<double> harmonicAverages(graph.getSize(), 0);

    for (VertexIndex vertex : graph)
        harmonicAverages[vertex] =
            getShortestPathHarmonicAverage(graph, vertex);

    return harmonicAverages;
}

template <typename EdgeLabel>
std::vector<double>
getBetweennessCentralities(const LabeledDirectedGraph<EdgeLabel> &graph,
                           bool normalizeWithGeodesicNumber) {
    size_t verticesNumber = graph.getSize();
    std::vector<double> betweennesses;
    betweennesses.resize(verticesNumber, 0);

    algorithms::MultiplePredecessors distancesPredecessors;
    std::list<std::list<VertexIndex>> currentGeodesics;
    for (const VertexIndex &i : graph) {
        distancesPredecessors = algorithms::findAllVertexPredecessors(graph, i);
        for (const VertexIndex &j : graph) {
            currentGeodesics =
                algorithms::findMultiplePathsToVertexFromPredecessors(
                    graph, i, j, distancesPredecessors);
            if (currentGeodesics.empty())
                continue; // vertices i and j are not in the same component

            for (auto &geodesic : currentGeodesics) {
                for (auto &vertexOnGeodesic : geodesic) {
                    if (vertexOnGeodesic != i && vertexOnGeodesic != j) {
                        if (normalizeWithGeodesicNumber)
                            betweennesses[vertexOnGeodesic] +=
                                1. / currentGeodesics.size();
                        else
                            betweennesses[vertexOnGeodesic] += 1;
                    }
                }
            }
        }
    }
    return betweennesses;
}

template <typename EdgeLabel>
std::vector<double>
getBetweennessCentralities(const LabeledUndirectedGraph<EdgeLabel> &graph,
                           bool normalizeWithGeodesicNumber) {
    size_t verticesNumber = graph.getSize();
    std::vector<double> betweennesses;
    betweennesses.resize(verticesNumber, 0);

    algorithms::MultiplePredecessors distancesPredecessors;
    std::list<std::list<VertexIndex>> currentGeodesics;
    for (const VertexIndex &i : graph) {
        distancesPredecessors = algorithms::findAllVertexPredecessors(graph, i);
        for (VertexIndex j = 0; j < i; j++) {
            currentGeodesics =
                algorithms::findMultiplePathsToVertexFromPredecessors(
                    graph, i, j, distancesPredecessors);
            if (currentGeodesics.empty())
                continue; // vertices i and j are not in the same component

            for (auto &geodesic : currentGeodesics) {
                for (auto &vertexOnGeodesic : geodesic) {
                    if (vertexOnGeodesic != i && vertexOnGeodesic != j) {
                        if (normalizeWithGeodesicNumber)
                            betweennesses[vertexOnGeodesic] +=
                                1. / currentGeodesics.size();
                        else
                            betweennesses[vertexOnGeodesic] += 1;
                    }
                }
            }
        }
    }
    return betweennesses;
}

} // namespace metrics
} // namespace BaseGraph

#endif
