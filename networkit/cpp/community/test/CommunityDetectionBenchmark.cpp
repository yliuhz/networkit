/*
 * CommunityDetectionBenchmark.h
 *
 *  Created on: 16.05.2014
 *      Author: Klara Reichard (klara.reichard@gmail.com), Marvin Ritter (marvin.ritter@gmail.com)
 */

#include <gtest/gtest.h>

#include <functional>
#include <map>

#include <networkit/auxiliary/Timer.hpp>
#include <networkit/centrality/Betweenness.hpp>
#include <networkit/centrality/PageRank.hpp>
#include <networkit/community/Modularity.hpp>
#include <networkit/community/PLM.hpp>
#include <networkit/community/PLP.hpp>
#include <networkit/structures/Partition.hpp>

#include <networkit/graph/Graph.hpp>
#include <networkit/io/METISGraphReader.hpp>

namespace NetworKit {

class CommunityDetectionBenchmark : public testing::Test {
public:
    virtual ~CommunityDetectionBenchmark() = default;

protected:
    METISGraphReader metisReader;
};

constexpr int runs = 1;

TEST_F(CommunityDetectionBenchmark, benchClusteringAlgos) {
    Aux::Timer timer;
    Modularity mod;

    // std::string graph = "../graphs/in-2004.graph";
    // std::string graph = "../graphs/uk-2002.graph";
    // std::string graph = "../graphs/uk-2007-05.graph";
    std::string graph = "/home/yliumh/github/networkit/input/polblogs.graph";

    DEBUG("Reading graph file ", graph.c_str(), " ...");
    timer.start();
    const Graph G = this->metisReader.read(graph);
    timer.stop();
    DEBUG("Reading graph took ", timer.elapsedMilliseconds() / 1000.0, "s");

    for (int r = 0; r < runs; r++) {
        Graph Gcopy = G;
        PLP algo(Gcopy);

        timer.start();
        algo.run();
        Partition zeta = algo.getPartition();
        timer.stop();

        auto communitySizes = zeta.subsetSizes();

        INFO("Parallel Label Propagation on ", graph.c_str(), ": ",
             (timer.elapsedMilliseconds() / 1000.0), "s,\t#communities: ", zeta.numberOfSubsets(),
             ",\tmodularity: ", mod.getQuality(zeta, G));
    }

    for (int r = 0; r < runs; r++) {
        Graph Gcopy = G;
        PLM algo(Gcopy);

        timer.start();
        algo.run();
        Partition zeta = algo.getPartition();
        timer.stop();

        auto communitySizes = zeta.subsetSizes();

        INFO("Parallel Louvain on ", graph.c_str(), ": ", (timer.elapsedMilliseconds() / 1000.0),
             "s,\t#communities: ", zeta.numberOfSubsets(),
             ",\tmodularity: ", mod.getQuality(zeta, G));
    }
}

TEST_F(CommunityDetectionBenchmark, benchPageRankCentrality) {
    Aux::Timer timer;

    // std::string graph = "../graphs/uk-2002.graph";
    std::string graph = "input/polblogs.graph";

    const Graph G = this->metisReader.read(graph);

    for (int r = 0; r < runs; r++) {
        PageRank cen(G, 1e-6);

        timer.start();
        cen.run();
        timer.stop();
        auto ranking = cen.ranking();

        INFO("Page Rank Centrality on ", graph.c_str(), ": ",
             (timer.elapsedMilliseconds() / 1000.0), "s,\t ranking: [(", ranking[0].first, ": ",
             ranking[0].second, "), (", ranking[1].first, ": ", ranking[1].second, ") ...]");
    }
}

TEST_F(CommunityDetectionBenchmark, benchBetweennessCentrality) {
    Aux::Timer timer;

    // std::string graph = "../graphs/cond-mat-2005.graph";
    std::string graph = "input/polblogs.graph";

    const Graph G = this->metisReader.read(graph);

    for (int r = 0; r < runs; r++) {
        Betweenness cen(G);

        timer.start();
        cen.run();
        timer.stop();
        auto ranking = cen.ranking();

        INFO("Betweenness Centrality on ", graph.c_str(), ": ",
             (timer.elapsedMilliseconds() / 1000.0), "s,\t ranking: [(", ranking[0].first, ": ",
             ranking[0].second, "), (", ranking[1].first, ": ", ranking[1].second, ") ...]");
    }
}

TEST_F(CommunityDetectionBenchmark, advancedNodeMoving) {
    Aux::Timer timer;
    Modularity mod;

    std::string graph = "/home/yliumh/github/networkit/input/polblogs.graph";

    DEBUG("Reading graph file ", graph.c_str(), " ...");
    timer.start();
    const Graph G = this->metisReader.read(graph);
    timer.stop();
    DEBUG("Reading graph took ", timer.elapsedMilliseconds() / 1000.0, "s");

    for (int r = 0; r < runs; r++) {
        DEBUG("What's up??");
        Graph Gcopy = G;
        PLM algo(Gcopy, false, 1.0, "none", 32, true, true, "ps");
        DEBUG("have you been here?");

        timer.start();
        algo.run();
        Partition zeta = algo.getPartition();
        timer.stop();

        auto communitySizes = zeta.subsetSizes();

        INFO("Parallel Louvain on ", graph.c_str(), ": ", (timer.elapsedMilliseconds() / 1000.0),
             "s,\t#communities: ", zeta.numberOfSubsets(),
             ",\tmodularity: ", mod.getQuality(zeta, G));
        
        auto visitCountSum = algo.getCount();
        int vcs = 0;
        for (const auto & vc : visitCountSum) {
            vcs += vc;
        }
        INFO("visit", visitCountSum);
        INFO("#Total visit= ", vcs);

        auto timing = algo.getTiming();
        INFO("Timing", timing);

        auto iter = algo.getIter();
        INFO("#Iter", iter);
    }
}


TEST_F(CommunityDetectionBenchmark, customDistribution) {
    
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


    std::random_device rd;
    std::mt19937 rng(rd());  // Create a random number generator engine for each thread

    std::vector<double> probabilities = {0.1, 0.1, 0.8};  // Custom probabilities
    CustomDistribution customDistribution(probabilities);

    for (int i = 0; i < 10; ++i) {
        int random_integer = customDistribution(rng);
        std::cout << "Random integer: " << random_integer << std::endl;
    }
}


TEST_F(CommunityDetectionBenchmark, softmax) {
    
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

    std::vector<double> ps = {1,2,3,4,5,6};
    ps = softmax(ps);

    double sum = 0.0;
    for (auto p : ps){
        std::cout<<p<<std::endl;
        sum += p;
    }
    std::cout<<"sum="<<sum<<std::endl;



}



} /* namespace NetworKit */
