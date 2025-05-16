#include "gtest/gtest.h"
#include "BaseGraph/undirected_graph.hpp"
#include "BaseGraph/directed_graph.hpp"


class UndirectedHouseGraph: public::testing::Test{
    /*
     * (0)     (1)
     *  | \   / | \
     *  |  \ /  |  \
     *  |   X   |  (4)
     *  |  / \  |  /
     *  | /   \ | /
     * (2)-----(3)-----(5)
     *
     *      (6)
     */
    public:
        BaseGraph::UndirectedGraph graph = BaseGraph::UndirectedGraph(7);
        void SetUp() {
            graph.addEdge(0, 2);
            graph.addEdge(0, 3);
            graph.addEdge(1, 2);
            graph.addEdge(1, 3);
            graph.addEdge(1, 4);
            graph.addEdge(2, 3);
            graph.addEdge(3, 4);
            graph.addEdge(3, 5);
        }
};


class DirectedHouseGraph: public::testing::Test{
    /*
     * (0)_    (1)
     *  ||\   / | \
     *  |  \ /  |  V
     *  |   X   |  (4)
     *  |  / \  |  /
     *  V V   \ | V
     *  (2)---->(3)---->(5)
     *     <----
     *
     *      (6)
     */
    public:
        BaseGraph::DirectedGraph graph = BaseGraph::DirectedGraph(7);
        void SetUp() {
            graph.addEdge(0, 2);
            graph.addEdge(3, 0);
            graph.addEdge(1, 2);
            graph.addEdge(3, 1);
            graph.addEdge(1, 4);
            graph.addReciprocalEdge(2, 3);
            graph.addEdge(4, 3);
            graph.addEdge(3, 5);
        }
};


class TreeLikeGraph: public::testing::Test{
    /*
     *        (0)
     *       /   \
     *     (1)   (2)
     *    /   \ /   \
     *   (3)  (4)  (5)
     *      \  |  /
     *        (6)
     *         |
     *        (7)
     */
    public:
        BaseGraph::UndirectedGraph graph = BaseGraph::UndirectedGraph(8);
        void SetUp(){
            graph.addEdge(0, 1);
            graph.addEdge(0, 2);
            graph.addEdge(1, 3);
            graph.addEdge(1, 4);
            graph.addEdge(2, 4);
            graph.addEdge(2, 5);
            graph.addEdge(3, 6);
            graph.addEdge(4, 6);
            graph.addEdge(5, 6);
            graph.addEdge(6, 7);
        }
};

class ThreeComponentsGraph: public::testing::Test{
    /*
     *        (0)--(1)--(2)--(3)
     *
     *           (7)--(8)
     *           /  \
     *         (6)  (9)
     *        /   \
     *      (4)---(5)     (10)
     */
    public:
        BaseGraph::UndirectedGraph graph = BaseGraph::UndirectedGraph(11);
        void SetUp(){
            graph.addEdge(0, 1);
            graph.addEdge(1, 2);
            graph.addEdge(2, 3);

            graph.addEdge(4, 5);
            graph.addEdge(5, 6);
            graph.addEdge(6, 4);
            graph.addEdge(6, 7);
            graph.addEdge(7, 8);
            graph.addEdge(7, 9);
        }
};
