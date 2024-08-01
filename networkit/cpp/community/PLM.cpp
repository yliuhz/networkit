/*
 * MLPLM.cpp
 *
 *  Created on: 20.11.2013
 *      Author: cls
 */

#include <omp.h>
#include <sstream>
#include <utility>
#include <bitset>

#include <networkit/auxiliary/Log.hpp>
#include <networkit/auxiliary/SignalHandling.hpp>
#include <networkit/auxiliary/Timer.hpp>
#include <networkit/coarsening/ClusteringProjector.hpp>
#include <networkit/coarsening/ParallelPartitionCoarsening.hpp>
#include <networkit/community/PLM.hpp>

#include <networkit/graph/GraphTools.hpp>
#include <tuple>

namespace NetworKit {

PLM::PLM(const Graph &G, bool refine, double gamma, std::string par, count maxIter, bool turbo,
         bool recurse, std::string nm)
    : CommunityDetectionAlgorithm(G), parallelism(std::move(par)), refine(refine), gamma(gamma),
      maxIter(maxIter), turbo(turbo), recurse(recurse), nm(std::move(nm)) {}

PLM::PLM(const Graph &G, const PLM &other)
    : CommunityDetectionAlgorithm(G), parallelism(other.parallelism), refine(other.refine),
      gamma(other.gamma), maxIter(other.maxIter), turbo(other.turbo), recurse(other.recurse) {}

void PLM::run() {

    // std::cout << "I am Yue Liu!" << std::endl;
    // std::cout << "Num of threads: " << omp_get_num_threads() << " " << omp_get_max_threads() << std::endl;
    DEBUG("Entering PLM::run()");

    Aux::SignalHandler handler;

    count z = G->upperNodeIdBound();

    // init communities to singletons
    Partition zeta(z);
    zeta.allToSingletons();
    index o = zeta.upperBound();

    // init graph-dependent temporaries
    std::vector<double> volNode(z, 0.0);
    // $\omega(E)$
    edgeweight total = G->totalEdgeWeight();
    DEBUG("total edge weight: ", total);
    edgeweight divisor = (2 * total * total); // needed in modularity calculation

    G->parallelForNodes([&](node u) { // calculate and store volume of each node
        volNode[u] += G->weightedDegree(u);
        volNode[u] += G->weight(u, u); // consider self-loop twice
    });

    // init community-dependent temporaries
    std::vector<double> volCommunity(o, 0.0);
    zeta.parallelForEntries([&](node u, index C) { // set volume for all communities
        if (C != none)
            volCommunity[C] = volNode[u];
    });

    // first move phase
    bool moved = false;  // indicates whether any node has been moved in the last pass
    bool change = false; // indicates whether the communities have changed at all
    std::vector<count> visitCount(z, 0);

    // stores the affinity for each neighboring community (index), one vector per thread
    std::vector<std::vector<edgeweight>> turboAffinity;
    // stores the list of neighboring communities, one vector per thread
    std::vector<std::vector<index>> neigh_comm;

    if (turbo) {
        // initialize arrays for all threads only when actually needed
        if (this->parallelism != "none" && this->parallelism != "none randomized") {
            turboAffinity.resize(omp_get_max_threads());
            neigh_comm.resize(omp_get_max_threads());
            for (auto &it : turboAffinity) {
                // resize to maximum community id
                it.resize(zeta.upperBound());
            }
        } else { // initialize array only for first thread
            turboAffinity.emplace_back(zeta.upperBound());
            neigh_comm.emplace_back();
        }
    }

    // try to improve modularity by moving a node to neighboring clusters
    auto tryMove = [&](node u) {
        visitCount[u] += 1;

        // trying to move node u
        index tid = omp_get_thread_num();

        // collect edge weight to neighbor clusters
        std::map<index, edgeweight> affinity;

        if (turbo) {
            neigh_comm[tid].clear();
            // set all to -1 so we can see when we get to it the first time
            G->forNeighborsOf(u, [&](node v) { turboAffinity[tid][zeta[v]] = -1; });
            turboAffinity[tid][zeta[u]] = 0;
            G->forNeighborsOf(u, [&](node v, edgeweight weight) {
                if (u != v) {
                    index C = zeta[v];
                    if (turboAffinity[tid][C] == -1) {
                        // found the neighbor for the first time, initialize to 0 and add to list of
                        // neighboring communities
                        turboAffinity[tid][C] = 0;
                        neigh_comm[tid].push_back(C);
                    }
                    turboAffinity[tid][C] += weight;
                }
            });
        } else {
            G->forNeighborsOf(u, [&](node v, edgeweight weight) {
                if (u != v) {
                    index C = zeta[v];
                    affinity[C] += weight;
                }
            });
        }

        // sub-functions

        // $\vol(C \ {x})$ - volume of cluster C excluding node x
        auto volCommunityMinusNode = [&](index C, node x) {
            double volC = 0.0;
            double volN = 0.0;
            volC = volCommunity[C];
            if (zeta[x] == C) {
                volN = volNode[x];
                return volC - volN;
            } else {
                return volC;
            }
        };

        auto modGain = [&](node u, index C, index D, edgeweight affinityC, edgeweight affinityD) {
            double volN = 0.0;
            volN = volNode[u];
            double delta =
                (affinityD - affinityC) / total
                + this->gamma * ((volCommunityMinusNode(C, u) - volCommunityMinusNode(D, u)) * volN)
                      / divisor;
            return delta;
        };

        index best = none;
        index C = none;
        double deltaBest = -1;

        C = zeta[u];

        if (turbo) {
            edgeweight affinityC = turboAffinity[tid][C];

            for (index D : neigh_comm[tid]) {

                // consider only nodes in other clusters (and implicitly only nodes other than u)
                if (D != C) {
                    double delta = modGain(u, C, D, affinityC, turboAffinity[tid][D]);

                    if (delta > deltaBest) {
                        deltaBest = delta;
                        best = D;
                    }
                }
            }
        } else {
            edgeweight affinityC = affinity[C];

            for (auto it : affinity) {
                index D = it.first;
                // consider only nodes in other clusters (and implicitly only nodes other than u)
                if (D != C) {
                    double delta = modGain(u, C, D, affinityC, it.second);
                    if (delta > deltaBest) {
                        deltaBest = delta;
                        best = D;
                    }
                }
            }
        }

        if (deltaBest > 0) {                   // if modularity improvement possible
            assert(best != C && best != none); // do not "move" to original cluster

            zeta[u] = best; // move to best cluster
            // node u moved

            // mod update
            double volN = 0.0;
            volN = volNode[u];
// update the volume of the two clusters
#pragma omp atomic
            volCommunity[C] -= volN;
#pragma omp atomic
            volCommunity[best] += volN;

            moved = true; // change to clustering has been made
        }
    };


    // try to improve modularity by moving a node to neighboring clusters, using sampling
    auto tryMoveSampling = [&](node u) {
        visitCount[u] += 1;

        // trying to move node u
        index tid = omp_get_thread_num();

        // collect edge weight to neighbor clusters
        std::map<index, edgeweight> affinity;

        if (turbo) {
            neigh_comm[tid].clear();
            // set all to -1 so we can see when we get to it the first time
            G->forNeighborsOf(u, [&](node v) { turboAffinity[tid][zeta[v]] = -1; });
            turboAffinity[tid][zeta[u]] = 0;
            G->forNeighborsOf(u, [&](node v, edgeweight weight) {
                if (u != v) {
                    index C = zeta[v];
                    if (turboAffinity[tid][C] == -1) {
                        // found the neighbor for the first time, initialize to 0 and add to list of
                        // neighboring communities
                        turboAffinity[tid][C] = 0;
                        neigh_comm[tid].push_back(C);
                    }
                    turboAffinity[tid][C] += weight;
                }
            });
        } else {
            G->forNeighborsOf(u, [&](node v, edgeweight weight) {
                if (u != v) {
                    index C = zeta[v];
                    affinity[C] += weight;
                }
            });
        }

        // sub-functions

        // $\vol(C \ {x})$ - volume of cluster C excluding node x
        auto volCommunityMinusNode = [&](index C, node x) {
            double volC = 0.0;
            double volN = 0.0;
            volC = volCommunity[C];
            if (zeta[x] == C) {
                volN = volNode[x];
                return volC - volN;
            } else {
                return volC;
            }
        };

        auto modGain = [&](node u, index C, index D, edgeweight affinityC, edgeweight affinityD) {
            double volN = 0.0;
            volN = volNode[u];
            double delta =
                (affinityD - affinityC) / total
                + this->gamma * ((volCommunityMinusNode(C, u) - volCommunityMinusNode(D, u)) * volN)
                      / divisor;
            return delta;
        };

        index best = none;
        index C = none;
        double deltaBest = -1;

        C = zeta[u];

        if (turbo) {
            edgeweight affinityC = turboAffinity[tid][C];
            node neighbor = NetworKit::GraphTools::randomNeighbor(*G, u);

            if (neighbor != none && neighbor != u) {
                index D = zeta[neighbor];
                double delta = modGain(u, C, D, affinityC, turboAffinity[tid][D]);
                deltaBest = delta;
                best = D;
            }

        } else {
            edgeweight affinityC = affinity[C];

            // for (auto it : affinity) {
            //     index D = it.first;
            //     // consider only nodes in other clusters (and implicitly only nodes other than u)
            //     if (D != C) {
            //         double delta = modGain(u, C, D, affinityC, it.second);
            //         if (delta > deltaBest) {
            //             deltaBest = delta;
            //             best = D;
            //         }
            //     }
            // }

            // std::vector<node> neighbor_nodes;
            // G->forNeighborsOf(u, [&](node v, edgeweight weight) {
            //     if (u != v) { // Only choose other nodes
            //         neighbor_nodes.push_back(v);
            //     }
            // });

            node neighbor = NetworKit::GraphTools::randomNeighbor(*G, u);

            if (neighbor != none && neighbor != u) {
                // std::random_device rd;
                // std::mt19937 rng(rd());  // Create a random number generator engine for each thread

                // std::uniform_int_distribution<node> distribution(0, neighbor_nodes.size()-1);  // Define the range of random integers
                // node neighbor_index = distribution(rng);
                // node neighbor = neighbor_nodes[neighbor_index];

                index D = zeta[neighbor];
                edgeweight affinityD = affinity[D];
                double delta = modGain(u, C, D, affinityC, affinityD);
                deltaBest = delta;
                best = D;
            }
        }

        if (deltaBest > 0) {                   // if modularity improvement possible
            assert(best != C && best != none); // do not "move" to original cluster

            zeta[u] = best; // move to best cluster
            // node u moved

            // mod update
            double volN = 0.0;
            volN = volNode[u];
// update the volume of the two clusters
#pragma omp atomic
            volCommunity[C] -= volN;
#pragma omp atomic
            volCommunity[best] += volN;

            moved = true; // change to clustering has been made
        }
    };


    // try to improve modularity by moving a node to neighboring clusters, using pruning
    std::vector<bool> shouldRun(z, true);
    std::vector<std::vector<int>> shouldRunNext(omp_get_max_threads()); 
    auto tryMovePruning = [&](node u) {
        if (!shouldRun[u]) return;
        shouldRun[u] = false;
        visitCount[u] += 1;

        // trying to move node u
        index tid = omp_get_thread_num();
        
        // collect edge weight to neighbor clusters
        std::map<index, edgeweight> affinity;

        if (turbo) {
            neigh_comm[tid].clear();
            // set all to -1 so we can see when we get to it the first time
            G->forNeighborsOf(u, [&](node v) { turboAffinity[tid][zeta[v]] = -1; });
            turboAffinity[tid][zeta[u]] = 0;
            G->forNeighborsOf(u, [&](node v, edgeweight weight) {
                if (u != v) {
                    index C = zeta[v];
                    if (turboAffinity[tid][C] == -1) {
                        // found the neighbor for the first time, initialize to 0 and add to list of
                        // neighboring communities
                        turboAffinity[tid][C] = 0;
                        neigh_comm[tid].push_back(C);
                    }
                    turboAffinity[tid][C] += weight;
                }
            });
        } else {
            G->forNeighborsOf(u, [&](node v, edgeweight weight) {
                if (u != v) {
                    index C = zeta[v];
                    affinity[C] += weight;
                }
            });
        }

        // sub-functions

        // $\vol(C \ {x})$ - volume of cluster C excluding node x
        auto volCommunityMinusNode = [&](index C, node x) {
            double volC = 0.0;
            double volN = 0.0;
            volC = volCommunity[C];
            if (zeta[x] == C) {
                volN = volNode[x];
                return volC - volN;
            } else {
                return volC;
            }
        };

        auto modGain = [&](node u, index C, index D, edgeweight affinityC, edgeweight affinityD) {
            double volN = 0.0;
            volN = volNode[u];
            double delta =
                (affinityD - affinityC) / total
                + this->gamma * ((volCommunityMinusNode(C, u) - volCommunityMinusNode(D, u)) * volN)
                      / divisor;
            return delta;
        };

        index best = none;
        index C = none;
        double deltaBest = -1;

        C = zeta[u];

        if (turbo) {
            edgeweight affinityC = turboAffinity[tid][C];

            for (index D : neigh_comm[tid]) {

                // consider only nodes in other clusters (and implicitly only nodes other than u)
                if (D != C) {
                    double delta = modGain(u, C, D, affinityC, turboAffinity[tid][D]);

                    if (delta > deltaBest) {
                        deltaBest = delta;
                        best = D;
                    }
                }
            }
        } else {
            edgeweight affinityC = affinity[C];

            for (auto it : affinity) {
                index D = it.first;
                // consider only nodes in other clusters (and implicitly only nodes other than u)
                if (D != C) {
                    double delta = modGain(u, C, D, affinityC, it.second);
                    if (delta > deltaBest) {
                        deltaBest = delta;
                        best = D;
                    }
                }
            }
        }

        if (deltaBest > 0) {                   // if modularity improvement possible
            assert(best != C && best != none); // do not "move" to original cluster

            zeta[u] = best; // move to best cluster
            // node u moved

            // mod update
            double volN = 0.0;
            volN = volNode[u];
// update the volume of the two clusters
#pragma omp atomic
            volCommunity[C] -= volN;
#pragma omp atomic
            volCommunity[best] += volN;

            moved = true; // change to clustering has been made

            // Wake its neighbor threads in the next iteration
            G->forNeighborsOf(u, [&](node v, edgeweight weight) {
                if (u != v && zeta[v] != best) {
                    shouldRunNext[tid].push_back(v); // Avoid lock
                }
            });
        }
    };

    // try to improve modularity by moving a node to neighboring clusters, using weighted sampling
    auto tryMoveSamplingWeighted = [&](node u) {
        visitCount[u] += 1;

        // trying to move node u
        index tid = omp_get_thread_num();

        // collect edge weight to neighbor clusters
        std::map<index, edgeweight> affinity;

        if (turbo) {
            neigh_comm[tid].clear();
            // set all to -1 so we can see when we get to it the first time
            G->forNeighborsOf(u, [&](node v) { turboAffinity[tid][zeta[v]] = -1; });
            turboAffinity[tid][zeta[u]] = 0;
            G->forNeighborsOf(u, [&](node v, edgeweight weight) {
                if (u != v) {
                    index C = zeta[v];
                    if (turboAffinity[tid][C] == -1) {
                        // found the neighbor for the first time, initialize to 0 and add to list of
                        // neighboring communities
                        turboAffinity[tid][C] = 0;
                        neigh_comm[tid].push_back(C);
                    }
                    turboAffinity[tid][C] += weight;
                }
            });
        } else {
            G->forNeighborsOf(u, [&](node v, edgeweight weight) {
                if (u != v) {
                    index C = zeta[v];
                    affinity[C] += weight;
                }
            });
        }

        // sub-functions

        // $\vol(C \ {x})$ - volume of cluster C excluding node x
        auto volCommunityMinusNode = [&](index C, node x) {
            double volC = 0.0;
            double volN = 0.0;
            volC = volCommunity[C];
            if (zeta[x] == C) {
                volN = volNode[x];
                return volC - volN;
            } else {
                return volC;
            }
        };

        auto modGain = [&](node u, index C, index D, edgeweight affinityC, edgeweight affinityD) {
            double volN = 0.0;
            volN = volNode[u];
            double delta =
                (affinityD - affinityC) / total
                + this->gamma * ((volCommunityMinusNode(C, u) - volCommunityMinusNode(D, u)) * volN)
                      / divisor;
            return delta;
        };

        index best = none;
        index C = none;
        double deltaBest = -1;

        C = zeta[u];

        std::random_device rd;
        std::mt19937 rng(rd());  // Create a random number generator engine for each thread
        // Custom distribution based on a probability distribution
        class CustomDistribution {
        public:
            explicit CustomDistribution(const std::vector<double>& probabilities)
                : probabilities_(probabilities), distribution_(0.0, 1.0) {
                std::partial_sum(probabilities_.begin(), probabilities_.end(), std::back_inserter(cumulativeProbabilities_));
            }

            int operator()(std::mt19937& rng) {
                double randomValue = distribution_(rng);

                auto it = std::lower_bound(cumulativeProbabilities_.begin(), cumulativeProbabilities_.end(), randomValue);
                if (it != cumulativeProbabilities_.end()) {
                    return std::distance(cumulativeProbabilities_.begin(), it);
                }

                return -1;  // Error case
            }

        private:
            std::vector<double> probabilities_;
            std::vector<double> cumulativeProbabilities_;
            std::uniform_real_distribution<double> distribution_;
        };

        auto softmax = [&](const std::vector<double>& input) {
            double max_value = *max_element(input.begin(), input.end());

            std::vector<double> result;
            double sum = 0.0;

            // Compute exponentials and sum
            for (double value : input) {
                double expValue = std::exp(value-max_value);
                result.push_back(expValue);
                sum += expValue;
            }

            // Normalize by the sum
            for (double& value : result) {
                value /= sum;
            }

            return result;
        };

        if (turbo) {
            throw std::runtime_error("Not implemented turbo for sampling");

            // edgeweight affinityC = turboAffinity[tid][C];

            // for (index D : neigh_comm[tid]) {

            //     // consider only nodes in other clusters (and implicitly only nodes other than u)
            //     if (D != C) {
            //         double delta = modGain(u, C, D, affinityC, turboAffinity[tid][D]);

            //         if (delta > deltaBest) {
            //             deltaBest = delta;
            //             best = D;
            //         }
            //     }
            // }
        } else {
            edgeweight affinityC = affinity[C];
            affinity.erase(C); // Only move to other communities

            if (affinity.size() > 0) {
                std::vector<NetworKit::index> neigh_communities; // diff from neigh_comm in turbo mode
                std::vector<double> cust_dist; // Neighbor community distribution
                for (const auto & it : affinity){
                    neigh_communities.emplace_back(it.first);
                    cust_dist.emplace_back(it.second);
                }

                std::vector<double> ps = softmax(cust_dist);

                CustomDistribution customDistribution(ps);
                int community_index = customDistribution(rng); // random index in neigh_communities and cust_dist

                index D = neigh_communities[community_index];
                edgeweight affinityD = affinity[D];
                double delta = modGain(u, C, D, affinityC, affinityD);
                deltaBest = delta;
                best = D;
            }
            else{
                deltaBest = -1;
            }
            
            
        }

        if (deltaBest > 0) {                   // if modularity improvement possible
            assert(best != C && best != none); // do not "move" to original cluster

            zeta[u] = best; // move to best cluster
            // node u moved

            // mod update
            double volN = 0.0;
            volN = volNode[u];
// update the volume of the two clusters
#pragma omp atomic
            volCommunity[C] -= volN;
#pragma omp atomic
            volCommunity[best] += volN;

            moved = true; // change to clustering has been made
        }
    };


    // try to improve modularity by moving a node to neighboring clusters, using pruning
    // std::vector<bool> shouldRun2(z, true);
    // std::vector<std::vector<int>> shouldRunNext2(omp_get_max_threads()); 
    auto tryMovePS = [&](node u) {
        if (!shouldRun[u]) return;
        shouldRun[u] = false;
        visitCount[u] += 1;

        // trying to move node u
        index tid = omp_get_thread_num();
        
        // collect edge weight to neighbor clusters
        std::map<index, edgeweight> affinity;

        if (turbo) {
            neigh_comm[tid].clear();
            // set all to -1 so we can see when we get to it the first time
            G->forNeighborsOf(u, [&](node v) { turboAffinity[tid][zeta[v]] = -1; });
            turboAffinity[tid][zeta[u]] = 0;
            G->forNeighborsOf(u, [&](node v, edgeweight weight) {
                if (u != v) {
                    index C = zeta[v];
                    if (turboAffinity[tid][C] == -1) {
                        // found the neighbor for the first time, initialize to 0 and add to list of
                        // neighboring communities
                        turboAffinity[tid][C] = 0;
                        neigh_comm[tid].push_back(C);
                    }
                    turboAffinity[tid][C] += weight;
                }
            });
        } else {
            G->forNeighborsOf(u, [&](node v, edgeweight weight) {
                if (u != v) {
                    index C = zeta[v];
                    affinity[C] += weight;
                }
            });
        }

        // sub-functions

        // $\vol(C \ {x})$ - volume of cluster C excluding node x
        auto volCommunityMinusNode = [&](index C, node x) {
            double volC = 0.0;
            double volN = 0.0;
            volC = volCommunity[C];
            if (zeta[x] == C) {
                volN = volNode[x];
                return volC - volN;
            } else {
                return volC;
            }
        };

        auto modGain = [&](node u, index C, index D, edgeweight affinityC, edgeweight affinityD) {
            double volN = 0.0;
            volN = volNode[u];
            double delta =
                (affinityD - affinityC) / total
                + this->gamma * ((volCommunityMinusNode(C, u) - volCommunityMinusNode(D, u)) * volN)
                      / divisor;
            return delta;
        };

        index best = none;
        index C = none;
        double deltaBest = -1;

        C = zeta[u];

        if (turbo) {
            edgeweight affinityC = turboAffinity[tid][C];
            node neighbor = NetworKit::GraphTools::randomNeighbor(*G, u);

            if (neighbor != none && neighbor != u) {
                index D = zeta[neighbor];
                double delta = modGain(u, C, D, affinityC, turboAffinity[tid][D]);
                deltaBest = delta;
                best = D;
            }

        } else {
            edgeweight affinityC = affinity[C];
            node neighbor = NetworKit::GraphTools::randomNeighbor(*G, u);
            if (neighbor != none && neighbor != u) {
                index D = zeta[neighbor];
                edgeweight affinityD = affinity[D];
                double delta = modGain(u, C, D, affinityC, affinityD);
                deltaBest = delta;
                best = D;
            }
        }

        if (deltaBest > 0) {                   // if modularity improvement possible
            assert(best != C && best != none); // do not "move" to original cluster

            zeta[u] = best; // move to best cluster
            // node u moved

            // mod update
            double volN = 0.0;
            volN = volNode[u];
// update the volume of the two clusters
#pragma omp atomic
            volCommunity[C] -= volN;
#pragma omp atomic
            volCommunity[best] += volN;

            moved = true; // change to clustering has been made

            // Wake its neighbor threads in the next iteration
            G->forNeighborsOf(u, [&](node v, edgeweight weight) {
                if (u != v && zeta[v] != best) {
                    shouldRunNext[tid].push_back(v); // Avoid lock
                }
            });
        }
    };



    // performs node moves
    DEBUG("Entering movePhase");
    auto movePhase = [&]() {
        count iter = 0;
        DEBUG("iter:", iter);
        do {
            moved = false;
            // apply node movement according to parallelization strategy
            if (this->nm == "all") {
                if (this->parallelism == "none") {
                    G->forNodes(tryMove);
                } else if (this->parallelism == "simple") {
                    G->parallelForNodes(tryMove);
                } else if (this->parallelism == "balanced") {
                    G->balancedParallelForNodes(tryMove);
                } else if (this->parallelism == "none randomized") {
                    G->forNodesInRandomOrder(tryMove);
                } else {
                    ERROR("unknown parallelization strategy: ", this->parallelism);
                    throw std::runtime_error("unknown parallelization strategy");
                }
            }
            else if (this->nm == "unweighted") {
                if (this->parallelism == "none") {
                    G->forNodes(tryMoveSampling);
                } else if (this->parallelism == "simple") {
                    G->parallelForNodes(tryMoveSampling);
                } else if (this->parallelism == "balanced") {
                    G->balancedParallelForNodes(tryMoveSampling);
                } else if (this->parallelism == "none randomized") {
                    G->forNodesInRandomOrder(tryMoveSampling);
                } else {
                    ERROR("unknown parallelization strategy: ", this->parallelism);
                    throw std::runtime_error("unknown parallelization strategy");
                }
            }
            else if (this->nm == "queue") {
                if (iter > 0) {
#pragma omp parallel for schedule(guided)
                    for (index tid = 0; tid < omp_get_max_threads(); tid++) { // for all threads
                        for (const auto u : shouldRunNext[tid]) {
                            shouldRun[u] = true;
                        }
                    }
                    shouldRunNext.clear();
                    shouldRunNext.resize(omp_get_max_threads());
                }
                DEBUG("About to parallel");

                if (this->parallelism == "none") {
                    G->forNodes(tryMovePruning);
                } else if (this->parallelism == "simple") {
                    G->parallelForNodes(tryMovePruning);
                } else if (this->parallelism == "balanced") {
                    G->balancedParallelForNodes(tryMovePruning);
                } else if (this->parallelism == "none randomized") {
                    G->forNodesInRandomOrder(tryMovePruning);
                } else {
                    ERROR("unknown parallelization strategy: ", this->parallelism);
                    throw std::runtime_error("unknown parallelization strategy");
                }
            }
            else if (this->nm == "weighted") {
                if (this->parallelism == "none") {
                    G->forNodes(tryMoveSamplingWeighted);
                } else if (this->parallelism == "simple") {
                    G->parallelForNodes(tryMoveSamplingWeighted);
                } else if (this->parallelism == "balanced") {
                    G->balancedParallelForNodes(tryMoveSamplingWeighted);
                } else if (this->parallelism == "none randomized") {
                    G->forNodesInRandomOrder(tryMoveSamplingWeighted);
                } else {
                    ERROR("unknown parallelization strategy: ", this->parallelism);
                    throw std::runtime_error("unknown parallelization strategy");
                }
            }
            else if (this->nm == "ps") {
                if (iter > 0) {
#pragma omp parallel for schedule(guided)
                    for (index tid = 0; tid < omp_get_max_threads(); tid++) { // for all threads
                        for (const auto u : shouldRunNext[tid]) {
                            shouldRun[u] = true;
                        }
                    }
                    shouldRunNext.clear();
                    shouldRunNext.resize(omp_get_max_threads());
                }
                DEBUG("About to parallel");

                if (this->parallelism == "none") {
                    G->forNodes(tryMovePS);
                } else if (this->parallelism == "simple") {
                    G->parallelForNodes(tryMovePS);
                } else if (this->parallelism == "balanced") {
                    G->balancedParallelForNodes(tryMovePS);
                } else if (this->parallelism == "none randomized") {
                    G->forNodesInRandomOrder(tryMovePS);
                } else {
                    ERROR("unknown parallelization strategy: ", this->parallelism);
                    throw std::runtime_error("unknown parallelization strategy");
                }
            }
            
            if (moved)
                change = true;

            if (iter == maxIter) {
                WARN("move phase aborted after ", maxIter, " iterations");
            }
            iter += 1;
        } while (moved && (iter <= maxIter) && handler.isRunning());
        DEBUG("iterations in move phase: ", iter);
        iterCount.push_back(iter);
    };
    handler.assureRunning();
    // first move phase
    Aux::Timer timer;
    timer.start();

    movePhase();

    timer.stop();
    timing["move"].push_back(timer.elapsedMilliseconds());
    count vcs = 0;
    for (const auto & vc : visitCount) vcs += vc;
    visitCountSum.push_back(vcs);
    handler.assureRunning();
    if (recurse && change) {
        DEBUG("nodes moved, so begin coarsening and recursive call");

        timer.start();

        // coarsen graph according to communities
        std::pair<Graph, std::vector<node>> coarsened = coarsen(*G, zeta);

        timer.stop();
        timing["coarsen"].push_back(timer.elapsedMilliseconds());

        PLM onCoarsened(coarsened.first, this->refine, this->gamma, this->parallelism,
                        this->maxIter, this->turbo);
        onCoarsened.run();
        Partition zetaCoarse = onCoarsened.getPartition();

        // get timings
        auto tim = onCoarsened.getTiming();
        for (count t : tim["move"]) {
            timing["move"].push_back(t);
        }
        for (count t : tim["coarsen"]) {
            timing["coarsen"].push_back(t);
        }
        for (count t : tim["refine"]) {
            timing["refine"].push_back(t);
        }

        // get countings
        auto visitCoun = onCoarsened.getCount();
        for (count t : visitCoun) {
            visitCountSum.push_back(t);
        }


        DEBUG("coarse graph has ", coarsened.first.numberOfNodes(), " nodes and ",
              coarsened.first.numberOfEdges(), " edges");
        // unpack communities in coarse graph onto fine graph
        zeta = prolong(coarsened.first, zetaCoarse, *G, coarsened.second);
        // refinement phase
        if (refine) {
            DEBUG("refinement phase");
            // reinit community-dependent temporaries
            o = zeta.upperBound();
            volCommunity.clear();
            volCommunity.resize(o, 0.0);
            zeta.parallelForEntries([&](node u, index C) { // set volume for all communities
                if (C != none) {
                    edgeweight volN = volNode[u];
#pragma omp atomic
                    volCommunity[C] += volN;
                }
            });
            // second move phase
            timer.start();

            visitCount.assign(z, 0);
            movePhase();

            timer.stop();
            timing["refine"].push_back(timer.elapsedMilliseconds());

            vcs = 0;
            for (const auto & vc : visitCount) vcs += vc;
            visitCountSum.push_back(vcs);
        }
    }
    result = std::move(zeta);
    hasRun = true;
}

std::pair<Graph, std::vector<node>> PLM::coarsen(const Graph &G, const Partition &zeta) {
    ParallelPartitionCoarsening parCoarsening(G, zeta);
    parCoarsening.run();
    return {parCoarsening.getCoarseGraph(), parCoarsening.getFineToCoarseNodeMapping()};
}

Partition PLM::prolong(const Graph &, const Partition &zetaCoarse, const Graph &Gfine,
                       std::vector<node> nodeToMetaNode) {
    Partition zetaFine(Gfine.upperNodeIdBound());
    zetaFine.setUpperBound(zetaCoarse.upperBound());

    Gfine.forNodes([&](node v) {
        node mv = nodeToMetaNode[v];
        index cv = zetaCoarse[mv];
        zetaFine[v] = cv;
    });

    return zetaFine;
}

const std::map<std::string, std::vector<count>> &PLM::getTiming() const {
    assureFinished();
    return timing;
}

const std::vector<count> &PLM::getCount() const {
    assureFinished();
    return visitCountSum;
}

const std::vector<count> &PLM::getIter() const {
    assureFinished();
    return iterCount;
}

//  zeta0 = Louvain(Gs[0])
Graph PLM::progressive(const Partition &zeta0) {
    if (Gs.size() < 1) return Graph(0); // Need call addKNNGraph to add graphs to Gs first.

    Graph G0 = Gs[0]; // undirected graph

    for (int i = 1; i < Gs.size(); i++) {
        bool accept = false;

        Graph Gi = Gs[i];
        Graph Gj = Gs[i-1];

        Graph dG = Graph(Gi.upperNodeIdBound(), Gi.isWeighted(), Gi.isDirected());

        Gi.forEdges([&](node u, node v, edgeweight w) {
            if (!Gj.hasEdge(u, v)) {  // NoQA
                dG.addEdge(u, v, w);
            }
        });

        if (dG.totalEdgeWeight() == 0) {
            INFO("P: Graphs in sequence should not be same");
            break;
        }

        INFO("P: total edge weight in dG: ", dG.totalEdgeWeight(), " ,", Gi.totalEdgeWeight()-Gj.totalEdgeWeight());


        std::pair<Graph, std::vector<node>> coarsened = coarsen(dG, zeta0);
        Graph dGA = coarsened.first;
        std::vector<node> nodeToSuperNode = coarsened.second;
        std::vector<std::unordered_set<node>> superNodeToNodes(dGA.upperNodeIdBound());
        dG.forNodes([&](node u) {
            node superId = nodeToSuperNode[u];
            superNodeToNodes[superId].insert(u);
        });


        //  Init information for tryMove
        Aux::SignalHandler handler;
        count z = dGA.upperNodeIdBound();
        // init communities to singletons
        Partition zeta(z);
        zeta.allToSingletons();
        index o = zeta.upperBound();
        // init graph-dependent temporaries
        std::vector<double> volNode(z, 0.0);
        // $\omega(E)$
        edgeweight total = dGA.totalEdgeWeight();
        INFO("P: total edge weight in dGA: ", total);
        INFO("P: #Nodes: ", z);
        
        edgeweight divisor = (2 * total * total); // needed in modularity calculation
        dGA.parallelForNodes([&](node u) { // calculate and store volume of each node
            volNode[u] += dGA.weightedDegree(u);
            volNode[u] += dGA.weight(u, u); // consider self-loop twice
        });
        // init community-dependent temporaries
        std::vector<double> volCommunity(o, 0.0);
        zeta.parallelForEntries([&](node u, index C) { // set volume for all communities
            if (C != none)
                volCommunity[C] = volNode[u];
        });
        // first move phase
        bool moved = false;  // indicates whether any node has been moved in the last pass
        bool change = false; // indicates whether the communities have changed at all
        // stores the list of neighboring communities, one vector per thread
        std::vector<std::vector<index>> neigh_comm;



        // try to improve modularity by moving a node to neighboring clusters
        std::vector<std::vector<std::tuple<node, node, edgeweight>>> insertEdges(omp_get_max_threads());
        auto tryMove = [&](node u) {
            // trying to move node u
            index tid = omp_get_thread_num();
            // collect edge weight to neighbor clusters
            std::map<index, edgeweight> affinity;
            dGA.forNeighborsOf(u, [&](node v, edgeweight weight) {
                if (u != v) {
                    index C = zeta[v];
                    affinity[C] += weight;
                }
            });
            // sub-functions
            // $\vol(C \ {x})$ - volume of cluster C excluding node x
            auto volCommunityMinusNode = [&](index C, node x) {
                double volC = 0.0;
                double volN = 0.0;
                volC = volCommunity[C];
                if (zeta[x] == C) {
                    volN = volNode[x];
                    return volC - volN;
                } else {
                    return volC;
                }
            };
            auto modGain = [&](node u, index C, index D, edgeweight affinityC, edgeweight affinityD) {
                double volN = 0.0;
                volN = volNode[u];
                double delta =
                    (affinityD - affinityC) / total
                    + this->gamma * ((volCommunityMinusNode(C, u) - volCommunityMinusNode(D, u)) * volN)
                        / divisor;
                return delta;
            };
            index best = none;
            index C = none;
            double deltaBest = -1;
            C = zeta[u];
            edgeweight affinityC = affinity[C];
            for (auto it : affinity) {
                index D = it.first;
                // consider only nodes in other clusters (and implicitly only nodes other than u)
                if (D != C) {
                    double delta = modGain(u, C, D, affinityC, it.second);
                    if (delta > deltaBest) {
                        deltaBest = delta;
                        best = D;
                    }
                }
            }

            if (deltaBest > 0 && best > C) {                   // if modularity improvement possible
                // assert(best != C && best != none); // do not "move" to original cluster

                moved = true; // change to clustering has been made

                // Wake its neighbor threads in the next iteration
                // G->forNeighborsOf(u, [&](node v, edgeweight weight) {
                //     if (u != v && zeta[v] != best) {
                //         shouldRunNext[tid].push_back(v); // Avoid lock
                //     }
                // });
                std::unordered_set<node> V1 = superNodeToNodes[C];
                std::unordered_set<node> V2 = superNodeToNodes[best];
                std::unordered_set<node> V3;
                std::set_union(std::begin(V1), std::end(V1),
                    std::begin(V2), std::end(V2),                  
                    std::inserter(V3, std::begin(V3)));
                

                Graph H1 = NetworKit::GraphTools::subgraphAndNeighborsFromNodes(dG, V1, false, false);
                Graph H2 = NetworKit::GraphTools::subgraphAndNeighborsFromNodes(dG, V2, false, false);
                Graph H = NetworKit::GraphTools::subgraphAndNeighborsFromNodes(dG, V3, false, false);
                
                
                // Homophilic check
                int m11 = H1.numberOfEdges();
                int m22 = H2.numberOfEdges();
                int m = H.numberOfEdges();

                // int m11 = H1.totalEdgeWeight();
                // int m22 = H2.totalEdgeWeight();
                // int m = H.totalEdgeWeight();

                
                int m12, m21;
                m12 = m21 = m - m11 - m22;

                double p11 = m11 * 1.0/ (m11 + m12);
                double p12 = m12 * 1.0/ (m11 + m12);
                double p21 = m21 * 1.0/ (m21 + m22);
                double p22 = m22 * 1.0/ (m21 + m22);

                // INFO("P: m: ", m11, ", ", m12, ", ", m21, ", ", m22);
                // INFO("P: p: ", p11, ", ", p12, ", ", p21, ", ", p22);

                if ((p11 > p12) && (p22 > p21)) {
                    H.forEdges([&](node u, node v, edgeweight w) {
                        insertEdges[tid].push_back(std::make_tuple(u, v, w));
                    });  // May insert duplicate edges
                    accept = true;
                };
            }
        };


        // performs node moves
        INFO("P: Entering movePhase");
        auto movePhase = [&]() {
            count iter = 0;
            count maxIter = 1;
            INFO("P: iter:", iter);
            do {
                moved = false;
                // apply node movement according to parallelization strategy
                if (this->parallelism == "none") {
                    dGA.forNodes(tryMove);
                } else if (this->parallelism == "simple") {
                    dGA.parallelForNodes(tryMove);
                } else if (this->parallelism == "balanced") {
                    dGA.balancedParallelForNodes(tryMove);
                } else if (this->parallelism == "none randomized") {
                    dGA.forNodesInRandomOrder(tryMove);
                } else {
                    ERROR("P: unknown parallelization strategy: ", this->parallelism);
                    throw std::runtime_error("unknown parallelization strategy");
                }
                
                if (moved)
                    change = true;
                if (iter == maxIter) {
                    WARN("P: move phase aborted after ", maxIter, " iterations");
                }
                iter += 1;
            } while (moved && (iter <= maxIter) && handler.isRunning());
            INFO("P: iterations in move phase: ", iter);
            iterCount.push_back(iter);
        };
        handler.assureRunning();
        // first move phase
        Aux::Timer timer;
        timer.start();
        movePhase();
        timer.stop();
        
        stopIter = i+1;

        if (!accept) {
            INFO("P: Break in ", i+1, "th graph");
            break;
        }
        else{
            INFO("P: Accept ", i+1, "th graph");
        }


        //  add insertEdges into G0
        std::vector<std::tuple<node, node, edgeweight>> insertEdgesAll;
        for (auto vs : insertEdges) {
            insertEdgesAll.insert(insertEdgesAll.end(), vs.begin(), vs.end());
        }
        INFO("#New edges: ", insertEdgesAll.size());
        
        // Graph tempG(G0.upperNodeIdBound());
        for (auto et : insertEdgesAll) {
            node u, v;
            edgeweight ew;
            u = std::get<0>(et);
            v = std::get<1>(et);
            ew = std::get<2>(et);

            // if (!tempG.hasEdge(u,v)) {
                // INFO("Adding edge ..");

                // tempG.addEdge(u,v);
            G0.increaseWeight(u, v, ew);
            // }
        }

    }

    return G0;
}

void PLM::addKNNGraph(const Graph &G){
    Gs.push_back(G);
}

const count PLM::getStopIter() const {
    assureFinished();
    return stopIter;
}

// void PLM::setDiffGraph(const Graph &G){
//     tempDG = G;
// }


//  zeta0 = Louvain(Gs[0])
Graph PLM::progressiveOnline(const Graph &dG, const Graph &GA0, const Partition &zeta0) {
        this->stop = 0;
        Graph retGA0 = GA0;
        bool accept = false;

        std::pair<Graph, std::vector<node>> coarsened = coarsen(dG, zeta0);
        Graph dGA = coarsened.first;
        // std::vector<node> nodeToSuperNode = coarsened.second;

        Graph dG_unweighted = Graph(dG, false, false);
        std::pair<Graph, std::vector<node>> coarsened_unweighted = coarsen(dG_unweighted, zeta0);
        Graph dGA_unweighted = coarsened_unweighted.first;

        //  Init information for tryMove
        Aux::SignalHandler handler;
        count z = dGA.upperNodeIdBound();
        // init communities to singletons
        Partition zeta(z);
        zeta.allToSingletons();
        index o = zeta.upperBound();
        // init graph-dependent temporaries
        std::vector<double> volNode(z, 0.0);
        // $\omega(E)$
        edgeweight total = dGA.totalEdgeWeight();
        INFO("P: total edge weight in dGA: ", total);
        INFO("P: #Nodes: ", z);
        
        edgeweight divisor = (2 * total * total); // needed in modularity calculation
        dGA.parallelForNodes([&](node u) { // calculate and store volume of each node
            volNode[u] += dGA.weightedDegree(u);
            volNode[u] += dGA.weight(u, u); // consider self-loop twice
        });
        // init community-dependent temporaries
        std::vector<double> volCommunity(o, 0.0);
        zeta.parallelForEntries([&](node u, index C) { // set volume for all communities
            if (C != none)
                volCommunity[C] = volNode[u];
        });
        // first move phase
        bool moved = false;  // indicates whether any node has been moved in the last pass
        bool change = false; // indicates whether the communities have changed at all
        // stores the list of neighboring communities, one vector per thread
        std::vector<std::vector<index>> neigh_comm;



        // try to improve modularity by moving a node to neighboring clusters
        std::vector<std::vector<std::tuple<node, node, edgeweight>>> insertEdges(omp_get_max_threads());
        auto tryMove = [&](node u) {
            // trying to move node u
            index tid = omp_get_thread_num();
            // collect edge weight to neighbor clusters
            std::map<index, edgeweight> affinity;
            dGA.forNeighborsOf(u, [&](node v, edgeweight weight) {
                if (u != v) {
                    index C = zeta[v];
                    affinity[C] += weight;
                }
            });
            // sub-functions
            // $\vol(C \ {x})$ - volume of cluster C excluding node x
            auto volCommunityMinusNode = [&](index C, node x) {
                double volC = 0.0;
                double volN = 0.0;
                volC = volCommunity[C];
                if (zeta[x] == C) {
                    volN = volNode[x];
                    return volC - volN;
                } else {
                    return volC;
                }
            };
            auto modGain = [&](node u, index C, index D, edgeweight affinityC, edgeweight affinityD) {
                double volN = 0.0;
                volN = volNode[u];
                double delta =
                    (affinityD - affinityC) / total
                    + this->gamma * ((volCommunityMinusNode(C, u) - volCommunityMinusNode(D, u)) * volN)
                        / divisor;
                return delta;
            };
            index best = none;
            index C = none;
            double deltaBest = -1;
            C = zeta[u];
            edgeweight affinityC = affinity[C];
            for (auto it : affinity) {
                index D = it.first;
                // consider only nodes in other clusters (and implicitly only nodes other than u)
                if (D != C) {
                    double delta = modGain(u, C, D, affinityC, it.second);
                    if (delta > deltaBest) {
                        deltaBest = delta;
                        best = D;
                    }
                }
            }

            if (deltaBest > 0 && best > C) {                   // if modularity improvement possible
                // assert(best != C && best != none); // do not "move" to original cluster

                moved = true; // change to clustering has been made

                // Wake its neighbor threads in the next iteration
                // G->forNeighborsOf(u, [&](node v, edgeweight weight) {
                //     if (u != v && zeta[v] != best) {
                //         shouldRunNext[tid].push_back(v); // Avoid lock
                //     }
                // });
                
                
                // Homophilic check
                // double m11 = dGA.weight(C, C);
                // double m22 = dGA.weight(best, best);
                // double m = m11 + m22 + dGA.weight(C, best);

                double m11 = dGA_unweighted.weight(C, C);
                double m22 = dGA_unweighted.weight(best, best);
                double m = m11 + m22 + dGA_unweighted.weight(C, best);


                double m12, m21;
                m12 = m21 = m - m11 - m22;

                double p11 = m11 * 1.0/ (m11 + m12);
                double p12 = m12 * 1.0/ (m11 + m12);
                double p21 = m21 * 1.0/ (m21 + m22);
                double p22 = m22 * 1.0/ (m21 + m22);

                // INFO("P: m: ", m11, ", ", m12, ", ", m21, ", ", m22);
                // INFO("P: p: ", p11, ", ", p12, ", ", p21, ", ", p22);

                if ((p11 > p12) && (p22 > p21)) {
                    insertEdges[tid].push_back(std::make_tuple(C, C, m11));
                    insertEdges[tid].push_back(std::make_tuple(best, best, m22));
                    insertEdges[tid].push_back(std::make_tuple(C, best, m12)); 
                    accept = true;
                };
            }
        };


        // performs node moves
        INFO("P: Entering movePhase");
        auto movePhase = [&]() {
            count iter = 0;
            count maxIter = 1;
            INFO("P: iter:", iter);
            do {
                moved = false;
                // apply node movement according to parallelization strategy
                if (this->parallelism == "none") {
                    dGA.forNodes(tryMove);
                } else if (this->parallelism == "simple") {
                    dGA.parallelForNodes(tryMove);
                } else if (this->parallelism == "balanced") {
                    dGA.balancedParallelForNodes(tryMove);
                } else if (this->parallelism == "none randomized") {
                    dGA.forNodesInRandomOrder(tryMove);
                } else {
                    ERROR("P: unknown parallelization strategy: ", this->parallelism);
                    throw std::runtime_error("unknown parallelization strategy");
                }
                
                if (moved)
                    change = true;
                if (iter == maxIter) {
                    WARN("P: move phase aborted after ", maxIter, " iterations");
                }
                iter += 1;
            } while (moved && (iter <= maxIter) && handler.isRunning());
            INFO("P: iterations in move phase: ", iter);
            iterCount.push_back(iter);
        };
        handler.assureRunning();
        // first move phase
        Aux::Timer timer;
        timer.start();
        movePhase();
        timer.stop();
        
        

        if (!accept) {
            this->stop = 1;
        }


        //  add insertEdges into G0
        std::vector<std::tuple<node, node, edgeweight>> insertEdgesAll;
        for (auto vs : insertEdges) {
            insertEdgesAll.insert(insertEdgesAll.end(), vs.begin(), vs.end());
        }
        
        // Graph tempG(G0.upperNodeIdBound());
        for (auto et : insertEdgesAll) {
            node u, v;
            edgeweight ew;
            u = std::get<0>(et);
            v = std::get<1>(et);
            ew = std::get<2>(et);

            // if (!tempG.hasEdge(u,v)) {
                // INFO("Adding edge ..");

                // tempG.addEdge(u,v);
            retGA0.increaseWeight(u, v, ew);
            // }
        }


    return retGA0;
}


Graph PLM::progressiveOnline_wo_hierarchy(const Graph &dG, const Graph &G0, const Partition &zeta0) {
        this->stop = 0;
        Graph retG0 = G0;
        bool accept = false;

        std::pair<Graph, std::vector<node>> coarsened = coarsen(dG, zeta0);
        Graph dGA = coarsened.first;
        std::vector<node> nodeToSuperNode = coarsened.second;
        std::vector<std::unordered_set<node>> superNodeToNodes(dGA.upperNodeIdBound());
        dG.forNodes([&](node u) {
            node superId = nodeToSuperNode[u];
            superNodeToNodes[superId].insert(u);
        });


        //  Init information for tryMove
        Aux::SignalHandler handler;
        count z = dGA.upperNodeIdBound();
        // init communities to singletons
        Partition zeta(z);
        zeta.allToSingletons();
        index o = zeta.upperBound();
        // init graph-dependent temporaries
        std::vector<double> volNode(z, 0.0);
        // $\omega(E)$
        edgeweight total = dGA.totalEdgeWeight();
        INFO("P: total edge weight in dGA: ", total);
        INFO("P: #Nodes: ", z);
        
        edgeweight divisor = (2 * total * total); // needed in modularity calculation
        dGA.parallelForNodes([&](node u) { // calculate and store volume of each node
            volNode[u] += dGA.weightedDegree(u);
            volNode[u] += dGA.weight(u, u); // consider self-loop twice
        });
        // init community-dependent temporaries
        std::vector<double> volCommunity(o, 0.0);
        zeta.parallelForEntries([&](node u, index C) { // set volume for all communities
            if (C != none)
                volCommunity[C] = volNode[u];
        });
        // first move phase
        bool moved = false;  // indicates whether any node has been moved in the last pass
        bool change = false; // indicates whether the communities have changed at all
        // stores the list of neighboring communities, one vector per thread
        std::vector<std::vector<index>> neigh_comm;

        // try to improve modularity by moving a node to neighboring clusters
        std::vector<std::vector<std::tuple<node, node, edgeweight>>> insertEdges(omp_get_max_threads());
        auto tryMove = [&](node u) {
            // trying to move node u
            index tid = omp_get_thread_num();
            // collect edge weight to neighbor clusters
            std::map<index, edgeweight> affinity;
            dGA.forNeighborsOf(u, [&](node v, edgeweight weight) {
                if (u != v) {
                    index C = zeta[v];
                    affinity[C] += weight;
                }
            });
            // sub-functions
            // $\vol(C \ {x})$ - volume of cluster C excluding node x
            auto volCommunityMinusNode = [&](index C, node x) {
                double volC = 0.0;
                double volN = 0.0;
                volC = volCommunity[C];
                if (zeta[x] == C) {
                    volN = volNode[x];
                    return volC - volN;
                } else {
                    return volC;
                }
            };
            auto modGain = [&](node u, index C, index D, edgeweight affinityC, edgeweight affinityD) {
                double volN = 0.0;
                volN = volNode[u];
                double delta =
                    (affinityD - affinityC) / total
                    + this->gamma * ((volCommunityMinusNode(C, u) - volCommunityMinusNode(D, u)) * volN)
                        / divisor;
                return delta;
            };
            index best = none;
            index C = none;
            double deltaBest = -1;
            C = zeta[u];
            edgeweight affinityC = affinity[C];
            for (auto it : affinity) {
                index D = it.first;
                // consider only nodes in other clusters (and implicitly only nodes other than u)
                if (D != C) {
                    double delta = modGain(u, C, D, affinityC, it.second);
                    if (delta > deltaBest) {
                        deltaBest = delta;
                        best = D;
                    }
                }
            }

            if (deltaBest > 0 && best > C) {                   // if modularity improvement possible
                // assert(best != C && best != none); // do not "move" to original cluster

                moved = true; // change to clustering has been made

                // Wake its neighbor threads in the next iteration
                // G->forNeighborsOf(u, [&](node v, edgeweight weight) {
                //     if (u != v && zeta[v] != best) {
                //         shouldRunNext[tid].push_back(v); // Avoid lock
                //     }
                // });
                std::unordered_set<node> V1 = superNodeToNodes[C];
                std::unordered_set<node> V2 = superNodeToNodes[best];
                std::unordered_set<node> V3;
                std::set_union(std::begin(V1), std::end(V1),
                    std::begin(V2), std::end(V2),                  
                    std::inserter(V3, std::begin(V3)));
                

                Graph H1 = NetworKit::GraphTools::subgraphAndNeighborsFromNodes(dG, V1, false, false);
                Graph H2 = NetworKit::GraphTools::subgraphAndNeighborsFromNodes(dG, V2, false, false);
                Graph H = NetworKit::GraphTools::subgraphAndNeighborsFromNodes(dG, V3, false, false);
                
                
                // Homophilic check
                double m11 = H1.numberOfEdges();
                double m22 = H2.numberOfEdges();
                double m = H.numberOfEdges();

                // double m11 = H1.totalEdgeWeight();
                // double m22 = H2.totalEdgeWeight();
                // double m = H.totalEdgeWeight();

                
                double m12, m21;
                m12 = m21 = m - m11 - m22;

                double p11 = m11 * 1.0/ (m11 + m12);
                double p12 = m12 * 1.0/ (m11 + m12);
                double p21 = m21 * 1.0/ (m21 + m22);
                double p22 = m22 * 1.0/ (m21 + m22);

                // INFO("P: m: ", m11, ", ", m12, ", ", m21, ", ", m22);
                // INFO("P: p: ", p11, ", ", p12, ", ", p21, ", ", p22);

                if ((p11 > p12) && (p22 > p21)) {
                    H.forEdges([&](node u, node v, edgeweight w) {
                        insertEdges[tid].push_back(std::make_tuple(u, v, w));
                    });  // May insert duplicate edges
                    accept = true;
                };
            }
        };


        // performs node moves
        INFO("P: Entering movePhase");
        auto movePhase = [&]() {
            count iter = 0;
            count maxIter = 1;
            INFO("P: iter:", iter);
            do {
                moved = false;
                // apply node movement according to parallelization strategy
                if (this->parallelism == "none") {
                    dGA.forNodes(tryMove);
                } else if (this->parallelism == "simple") {
                    dGA.parallelForNodes(tryMove);
                } else if (this->parallelism == "balanced") {
                    dGA.balancedParallelForNodes(tryMove);
                } else if (this->parallelism == "none randomized") {
                    dGA.forNodesInRandomOrder(tryMove);
                } else {
                    ERROR("P: unknown parallelization strategy: ", this->parallelism);
                    throw std::runtime_error("unknown parallelization strategy");
                }
                
                if (moved)
                    change = true;
                if (iter == maxIter) {
                    WARN("P: move phase aborted after ", maxIter, " iterations");
                }
                iter += 1;
            } while (moved && (iter <= maxIter) && handler.isRunning());
            INFO("P: iterations in move phase: ", iter);
            iterCount.push_back(iter);
        };
        handler.assureRunning();
        // first move phase
        Aux::Timer timer;
        timer.start();
        movePhase();
        timer.stop();

        if (!accept) {
            this->stop = 1;
        }


        //  add insertEdges into G0
        std::vector<std::tuple<node, node, edgeweight>> insertEdgesAll;
        for (auto vs : insertEdges) {
            insertEdgesAll.insert(insertEdgesAll.end(), vs.begin(), vs.end());
        }
        INFO("#New edges: ", insertEdgesAll.size());
        
        // Graph tempG(G0.upperNodeIdBound());
        for (auto et : insertEdgesAll) {
            node u, v;
            edgeweight ew;
            u = std::get<0>(et);
            v = std::get<1>(et);
            ew = std::get<2>(et);

            // if (!tempG.hasEdge(u,v)) {
                // INFO("Adding edge ..");

                // tempG.addEdge(u,v);
            retG0.increaseWeight(u, v, ew);
            // }
        }

    

    return retG0;
}



const count PLM::stopNow() const{
    return this->stop;
}


} /* namespace NetworKit */
