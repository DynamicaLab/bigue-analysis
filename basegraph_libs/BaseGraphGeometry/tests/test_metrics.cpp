#include "BaseGraph/extensions/geometry/metrics.hpp"
#include "fixtures.hpp"
#include "gtest/gtest.h"
#include <BaseGraph/undirected_graph.hpp>

using namespace std;
using namespace BaseGraph;

TEST(GreedyRouting, unsuccessful) {
    UndirectedGraph graph(4);
    graph.addEdge(0, 1);
    graph.addEdge(0, 2);
    graph.addEdge(1, 2);
    graph.addEdge(2, 3);
    std::vector<std::vector<double>> embedding = {{0}, {1}, {2}, {3}};
    std::vector<std::vector<double>> distanceMatrix = {
        {0, 2, 5, 5},
        {2, 0, 1, 4},
        {5, 1, 0, 5},
        {5, 4, 5, 0}
    };
    auto dist = [&distanceMatrix](const std::vector<double>& x1, const std::vector<double>& x2) {
        int i = x1[0];
        int j = x2[0];
        return distanceMatrix[i][j];
    };

    auto res = geometry::getGreedyRoutingScores(graph, embedding, dist);
    EXPECT_DOUBLE_EQ(res["successful"], 4./6);
}

TEST(GreedyRouting, correctStretch) {
    UndirectedGraph graph(5);
    graph.addEdge(0, 1);
    graph.addEdge(1, 2);
    graph.addEdge(2, 4);
    graph.addEdge(0, 3);
    graph.addEdge(3, 4);
    std::vector<std::vector<double>> embedding = {{0}, {1}, {2}, {3}, {4}};
    std::vector<std::vector<double>> distanceMatrix = {
        {0, 2, 2, 1, 7},
        {2, 0, 1, 1, 4},
        {2, 1, 0, 2, 3},
        {1, 1, 2, 0, 5},
        {7, 4, 3, 5, 0},
    };
    auto dist = [&distanceMatrix](const std::vector<double>& x1, const std::vector<double>& x2) {
        int i = x1[0];
        int j = x2[0];
        return distanceMatrix[i][j];
    };

    auto res = geometry::getGreedyRoutingScores(graph, embedding, dist);
    EXPECT_DOUBLE_EQ(res["stretch"], 11./10);
    EXPECT_DOUBLE_EQ(res["hyperstretch greedy"], (11+6./7+1.5)/10);
    EXPECT_DOUBLE_EQ(res["hyperstretch original"], (13+6./7+1.5)/10);
}
