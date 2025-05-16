#ifndef BASE_GRAPH_DIRECTED_GRAPH_METRICS_H
#define BASE_GRAPH_DIRECTED_GRAPH_METRICS_H

#include <array>
#include <map>
#include <unordered_set>
#include <vector>

#include "BaseGraph/directed_graph.hpp"

namespace BaseGraph {
namespace metrics {

template <typename T>
static std::list<T> intersection_of(const std::list<T> &, const std::list<T> &);
template <typename T>
static std::list<T> getUnionOfLists(const std::list<T> &, const std::list<T> &);

template <typename EdgeLabel>
double getDensity(const LabeledDirectedGraph<EdgeLabel> &graph) {
    size_t n = graph.getSize();
    return graph.getEdgeNumber() / ((double)n * (n - 1));
}

template <typename EdgeLabel>
double getReciprocity(const LabeledDirectedGraph<EdgeLabel> &graph) {
    size_t reciprocalEdgeNumber = 0;

    for (const VertexIndex &vertex : graph)
        for (const VertexIndex &neighbour : graph.getOutNeighbours(vertex))
            if (vertex < neighbour && graph.hasEdge(neighbour, vertex))
                reciprocalEdgeNumber += 2;

    return reciprocalEdgeNumber / (double)graph.getEdgeNumber();
}

template <typename EdgeLabel>
std::vector<size_t>
getReciprocalDegrees(const LabeledDirectedGraph<EdgeLabel> &graph) {
    std::vector<size_t> reciprocities(graph.getSize(), 0);

    for (const VertexIndex &vertex : graph) {
        for (const VertexIndex &neighbour : graph.getOutNeighbours(vertex)) {
            if (vertex < neighbour && graph.hasEdge(neighbour, vertex)) {
                reciprocities[vertex]++;
                reciprocities[neighbour]++;
            }
        }
    }

    return reciprocities;
}

template <typename EdgeLabel>
std::vector<double>
getJaccardReciprocities(const LabeledDirectedGraph<EdgeLabel> &graph,
                        const std::vector<size_t> &reciprocities,
                        const std::vector<size_t> &inDegrees) {
    if (reciprocities.size() != graph.getSize() ||
        inDegrees.size() != graph.getSize())
        throw std::logic_error(
            "The reciprocities and the in degrees must have the "
            "size of the graph.");

    std::vector<double> jaccardReciprocities(reciprocities.begin(),
                                             reciprocities.end());

    for (const VertexIndex &vertex : graph)
        jaccardReciprocities[vertex] /= inDegrees[vertex] +
                                        graph.getOutDegree(vertex) -
                                        (double)reciprocities[vertex];

    return jaccardReciprocities;
}

template <typename EdgeLabel>
std::vector<double>
getJaccardReciprocities(const LabeledDirectedGraph<EdgeLabel> &graph) {
    return getJaccardReciprocities(graph, getReciprocalDegrees(graph),
                                   graph.getInDegrees());
};

template <typename EdgeLabel>
std::vector<double>
getReciprocityRatios(const LabeledDirectedGraph<EdgeLabel> &graph,
                     const std::vector<size_t> &reciprocities,
                     const std::vector<size_t> &inDegrees) {
    if (reciprocities.size() != graph.getSize() ||
        inDegrees.size() != graph.getSize())
        throw std::logic_error(
            "The reciprocities and the in degrees must have the "
            "size of the graph.");

    std::vector<double> reciprocityRatios(reciprocities.begin(),
                                          reciprocities.end());

    for (const VertexIndex &vertex : graph)
        reciprocityRatios[vertex] *=
            (double)2 / (inDegrees[vertex] + graph.getOutDegree(vertex));

    return reciprocityRatios;
}

template <typename EdgeLabel>
std::vector<double>
getReciprocityRatios(const LabeledDirectedGraph<EdgeLabel> &graph) {
    return getReciprocityRatios(graph, getReciprocalDegrees(graph),
                                graph.getInDegrees());
};

template <typename EdgeLabel>
std::list<std::array<VertexIndex, 3>>
findAllDirectedTriangles(const LabeledDirectedGraph<EdgeLabel> &graph,
                         const LabeledDirectedGraph<EdgeLabel> &reversedGraph) {
    if (reversedGraph.getSize() != graph.getSize())
        throw std::logic_error(
            "The reversed graph must be the size of the graph.");
    std::list<std::array<VertexIndex, 3>> triangles;

    AdjacencyLists undirectedEdges(graph.getSize());

    for (const VertexIndex &vertex1 : graph)
        undirectedEdges[vertex1] =
            getUnionOfLists(graph.getOutNeighbours(vertex1),
                            reversedGraph.getOutNeighbours(vertex1));

    for (const VertexIndex &vertex1 : graph)
        for (const VertexIndex &vertex2 : undirectedEdges[vertex1])
            if (vertex1 < vertex2)
                for (const VertexIndex &vertex3 : intersection_of(
                         undirectedEdges[vertex1], undirectedEdges[vertex2]))
                    if (vertex2 < vertex3)
                        triangles.push_back({vertex1, vertex2, vertex3});

    return triangles;
}

template <typename EdgeLabel>
std::vector<double> getUndirectedLocalClusteringCoefficients(
    const LabeledDirectedGraph<EdgeLabel> &graph,
    const std::list<std::array<VertexIndex, 3>> &triangles,
    const LabeledDirectedGraph<EdgeLabel> &reversedGraph) {

    if (reversedGraph.getSize() != graph.getSize())
        throw std::logic_error(
            "The reversed graph must be the size of the graph.");

    std::vector<double> localClusteringCoefficients(graph.getSize(), 0);

    for (auto &triangle : triangles) {
        localClusteringCoefficients[triangle[0]]++;
        localClusteringCoefficients[triangle[1]]++;
        localClusteringCoefficients[triangle[2]]++;
    }

    size_t undirectedDegree;

    for (const VertexIndex &vertex : graph) {
        undirectedDegree =
            getUnionOfLists(graph.getOutNeighbours(vertex),
                            reversedGraph.getOutNeighbours(vertex))
                .size();
        if (undirectedDegree > 1)
            localClusteringCoefficients[vertex] /=
                undirectedDegree * (undirectedDegree - 1) / 2.;
    }
    return localClusteringCoefficients;
}

template <typename EdgeLabel>
std::vector<double> getUndirectedLocalClusteringCoefficients(
    const LabeledDirectedGraph<EdgeLabel> &graph,
    const LabeledDirectedGraph<EdgeLabel> &reversedGraph) {

    return getUndirectedLocalClusteringCoefficients(
        graph, findAllDirectedTriangles(graph, reversedGraph), reversedGraph);
}

template <typename EdgeLabel>
std::vector<double> getUndirectedLocalClusteringCoefficients(
    const LabeledDirectedGraph<EdgeLabel> &graph) {

    auto reversedGraph = graph.getReversedGraph();
    return getUndirectedLocalClusteringCoefficients(
        graph, findAllDirectedTriangles(graph, reversedGraph), reversedGraph);
}

template <typename EdgeLabel>
double getUndirectedGlobalClusteringCoefficient(
    const LabeledDirectedGraph<EdgeLabel> &graph,
    const std::list<std::array<VertexIndex, 3>> &triangles,
    const LabeledDirectedGraph<EdgeLabel> &reversedGraph) {
    if (reversedGraph.getSize() != graph.getSize())
        throw std::logic_error(
            "The reversed graph must be the size of the graph.");

    size_t totalDegree, localTriangles, triadNumber(0);
    for (VertexIndex &vertex : graph) {
        totalDegree =
            reversedGraph.getOutDegree(vertex) + graph.getOutDegree(vertex);

        localTriangles = getUnionOfLists(graph.getOutNeighbours(vertex),
                                         reversedGraph.getOutNeighbours(vertex))
                             .size();
        if (totalDegree > 1)
            triadNumber += localTriangles * (localTriangles - 1) / 2;
    }
    return (double)3 * triangles.size() / triadNumber;
}

template <typename EdgeLabel>
double getUndirectedGlobalClusteringCoefficient(
    const LabeledDirectedGraph<EdgeLabel> &graph) {

    auto reversedGraph = graph.getReversedGraph();
    return getUndirectedGlobalClusteringCoefficient(
        graph, findAllDirectedTriangles(graph, reversedGraph), reversedGraph);
}

template <typename EdgeLabel>
std::list<std::array<VertexIndex, 3>>
findAllDirectedTriangles(const LabeledDirectedGraph<EdgeLabel> &graph) {
    return findAllDirectedTriangles(graph, graph.getReversedGraph());
}

static const std::map<std::string, std::string> triangleEdgesToType = {
    {"-> -> -> ", "3cycle"},    {"<- <- <- ", "3cycle"},

    {"<- -> -> ", "3nocycle"},  {"-> <- -> ", "3nocycle"},
    {"-> -> <- ", "3nocycle"},  {"-> <- <- ", "3nocycle"},
    {"<- -> <- ", "3nocycle"},  {"<- <- -> ", "3nocycle"},

    {"<-> -> -> ", "4cycle"},   {"-> <-> -> ", "4cycle"},
    {"-> -> <-> ", "4cycle"},   {"<-> <- <- ", "4cycle"},
    {"<- <-> <- ", "4cycle"},   {"<- <- <-> ", "4cycle"},

    {"<-> -> <- ", "4outward"}, {"<- <-> -> ", "4outward"},
    {"-> <- <-> ", "4outward"},

    {"<-> <- -> ", "4inward"},  {"-> <-> <- ", "4inward"},
    {"<- -> <-> ", "4inward"},

    {"-> <-> <-> ", "5cycle"},  {"<-> -> <-> ", "5cycle"},
    {"<-> <-> -> ", "5cycle"},  {"<- <-> <-> ", "5cycle"},
    {"<-> <- <-> ", "5cycle"},  {"<-> <-> <- ", "5cycle"},

    {"<-> <-> <-> ", "6cycle"}};

static const std::list<std::string> triangleTypes = {
    "3cycle", "3nocycle", "4cycle", "4outward", "4inward", "5cycle", "6cycle"};

template <typename EdgeLabel>
std::map<std::string, std::size_t>
getTriangleSpectrum(const LabeledDirectedGraph<EdgeLabel> &graph,
                    const std::list<std::array<VertexIndex, 3>> &triangles) {
    std::map<std::string, size_t> triangleSpectrum;

    for (const std::string &triangleType : triangleTypes)
        triangleSpectrum[triangleType] = 0;

    std::string triangleEdgesRepresentation;
    std::list<Edge> triangleEdges = {{0, 1}, {1, 2}, {2, 0}};
    bool ij_isEdge;
    bool ji_isEdge;
    for (auto &triangle : triangles) {
        triangleEdgesRepresentation = "";

        for (auto &edge : triangleEdges) {
            ij_isEdge =
                graph.hasEdge(triangle[edge.first], triangle[edge.second]);
            ji_isEdge =
                graph.hasEdge(triangle[edge.second], triangle[edge.first]);

            if (ij_isEdge && !ji_isEdge)
                triangleEdgesRepresentation += "-> ";
            else if (!ij_isEdge && ji_isEdge)
                triangleEdgesRepresentation += "<- ";
            else // if edge ij and ji don't exist then the input isn't a
                 // complete triangle
                triangleEdgesRepresentation += "<-> ";
        }
        triangleSpectrum[triangleEdgesToType.at(triangleEdgesRepresentation)]++;
    }

    return triangleSpectrum;
}

template <typename EdgeLabel>
std::map<size_t, size_t>
getOutDegreeHistogram(const LabeledDirectedGraph<EdgeLabel> &graph) {
    std::map<size_t, size_t> outDegreeHistogram;

    auto outDegrees = graph.getOutDegrees();

    for (auto degree : outDegrees) {
        if (outDegreeHistogram.count(degree) == 0)
            outDegreeHistogram[degree] = 1;
        else
            outDegreeHistogram[degree] += 1;
    }

    return outDegreeHistogram;
}

template <typename EdgeLabel>
std::map<size_t, size_t>
getInDegreeHistogram(const LabeledDirectedGraph<EdgeLabel> &graph,
                     const std::vector<size_t> &inDegrees) {
    std::map<size_t, size_t> inDegreeHistogram;

    for (auto degree : inDegrees) {
        if (inDegreeHistogram.count(degree) == 0)
            inDegreeHistogram[degree] = 1;
        else
            inDegreeHistogram[degree] += 1;
    }

    return inDegreeHistogram;
}

template <typename EdgeLabel>
std::map<size_t, size_t>
getInDegreeHistogram(const LabeledDirectedGraph<EdgeLabel> &graph) {
    return getInDegreeHistogram(graph, graph.getInDegrees());
}

// From
// https://stackoverflow.com/questions/38993415/how-to-apply-the-intersection-between-two-lists-in-c
template <typename T>
static std::list<T> intersection_of(const std::list<T> &a,
                                    const std::list<T> &b) {
    std::list<T> rtn;
    std::unordered_multiset<T> st;
    std::for_each(a.begin(), a.end(), [&st](const T &k) { st.insert(k); });
    std::for_each(b.begin(), b.end(), [&st, &rtn](const T &k) {
        auto iter = st.find(k);
        if (iter != st.end()) {
            rtn.push_back(k);
            st.erase(iter);
        }
    });
    return rtn;
}

template <typename T>
static std::list<T> getUnionOfLists(const std::list<T> &list1,
                                    const std::list<T> &list2) {
    std::list<T> listUnion = list1;
    std::unordered_set<T> list1Set(list1.begin(), list1.end());

    for (auto &element : list2)
        if (list1Set.find(element) == list1Set.end())
            listUnion.push_back(element);

    return listUnion;
}

} // namespace metrics
} // namespace BaseGraph

#endif
