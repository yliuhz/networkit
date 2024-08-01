/*
 * PLM.hpp
 *
 *  Created on: 20.11.2013
 *      Author: cls
 */

#ifndef NETWORKIT_COMMUNITY_PLM_HPP_
#define NETWORKIT_COMMUNITY_PLM_HPP_

#include <networkit/community/CommunityDetectionAlgorithm.hpp>

namespace NetworKit {

/**
 * @ingroup community
 * Parallel Louvain Method - a multi-level modularity maximizer.
 */
class PLM final : public CommunityDetectionAlgorithm {

public:
    /**
     * @param[in] G input graph
     * @param[in] refine add a second move phase to refine the communities
     * @param[in] par parallelization strategy
     * @param[in] gammamulti-resolution modularity parameter:
     *            1.0 -> standard modularity
     *            0.0 -> one community
     *            2m -> singleton communities
     * @param[in] maxIter maximum number of iterations for move phase
     * @param[in] parallelCoarsening use parallel graph coarsening
     * @param[in] turbo faster but uses O(n) additional memory per thread
     * @param[in] recurse use recursive coarsening, see
     * http://journals.aps.org/pre/abstract/10.1103/PhysRevE.89.049902 for some explanations
     * (default: true)
     * @param[in] nm node moving strategy, in ["all", "unweighted", "weighted", "queue"]
     *
     */
    PLM(const Graph &G, bool refine = false, double gamma = 1.0, std::string par = "balanced",
        count maxIter = 32, bool turbo = true, bool recurse = true, std::string nm = "all");

    PLM(const Graph &G, const PLM &other);


    /**
     * Detect communities.
     */
    void run() override;

    /**
     * Coarsens a graph based on a given partition and returns both the coarsened graph and a
     * mapping for the nodes from fine to coarse.
     *
     * @param graph The input graph
     * @param zeta Partition of the graph, which represents the desired state of the coarsened graph
     * @return pair of coarsened graph and node-mappings from fine to coarse graph
     */
    static std::pair<Graph, std::vector<node>> coarsen(const Graph &G, const Partition &zeta);

    /**
     * Calculates a partition containing the mapping of node-id from a fine graph
     * to a cluster-id from partition based on a coarse graph.
     *
     * @param Gcoarse Coarsened graph
     * @param zetaCoarse Partition, which contains information about clusters in the coarsened graph
     * @param Gfine Fine graph
     * @param nodeToMetaNode mapping for node-id from fine to coarse graph
     * @return Partition, which contains the cluster-id in the coarse graph for every node from the
     * fine graph
     */
    static Partition prolong(const Graph &Gcoarse, const Partition &zetaCoarse, const Graph &Gfine,
                             std::vector<node> nodeToMetaNode);

    /**
     * Returns fine-grained running time measurements for algorithm engineering purposes.
     */
    const std::map<std::string, std::vector<count>> &getTiming() const;
    
    const std::vector<count> &getCount() const;

    const std::vector<count> &getIter() const;


    /**
     * Progressively build similarity graph
     */
    void addKNNGraph(const Graph &G);
    Graph progressive(const Partition &zeta0);
    const count getStopIter() const;
    // void setDiffGraph(const Graph &G);
    Graph progressiveOnline(const Graph &dG, const Graph &GA0, const Partition &zeta0);
    Graph progressiveOnline_wo_hierarchy(const Graph &dG, const Graph &G0, const Partition &zeta0);
    const count stopNow() const;

    
    

private:
    std::string parallelism;
    bool refine;
    double gamma = 1.0;
    count maxIter;
    bool turbo;
    bool recurse;
    std::string nm;
    std::map<std::string, std::vector<count>> timing; // fine-grained running time measurement
    std::vector<count> visitCountSum; // fine-grained node visiting count
    std::vector<count> iterCount;  

    std::vector<Graph> Gs; // set of knn graphs
    count stopIter = 1; // PGC breaks at ith iteration
    count stop = 0;
    // Graph tempDG; // differential knn graph
};

} /* namespace NetworKit */

#endif // NETWORKIT_COMMUNITY_PLM_HPP_
