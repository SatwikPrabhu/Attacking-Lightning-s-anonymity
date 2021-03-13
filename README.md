# Evaluation of Lightning's Anonymity

An attack that enables an intermediary to break the anonymity of the source and destination of a trannsaction in the Lightning network. 

This includes a simulator to simulate transactions using **LND**(https://github.com/lightningnetwork/lnd/blob/master/routing/pathfind.go), **c-Lightning**(https://github.com/ElementsProject/lightning/blob/f3159ec4acd1013427c292038b88071b868ab1ff/common/route.c) and **Eclair**(https://github.com/ACINQ/eclair/blob/master/eclair-core/src/main/scala/fr/acinq/eclair/router/Router.scala). 

All path finding algorithms are based on the Dijkstra's algorithm using a priority queue(https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm).

We modify Eclair to use a generalized version of Dijkstra's algorithm(https://en.wikipedia.org/wiki/K_shortest_path_routing) instead of using Yen's algorithm(https://en.wikipedia.org/wiki/Yen%27s_algorithm). We do not have code to simulate concurrent payments yet. 

The experiment is run on a snapshot of the Lightning Network obtained from https://ln.bigsun.xyz. The set of adversaries is a mixture of nodes with high centrality, low centrality and random nodes. The snapshot as well as the centralities of all nodes are found in data/Snapshot and data/Centrality respectively.

## Code Structure

*populate_graph.py* - Creates a payment graph from a snapshot of the Lightning Network.

*pathFind.py* - Implements Dijkstra and generalized Dijkstra(for 3 best paths) taking the cost function of either LND, c-Lightning or Eclair as argument.

*attack_mixed.py* - Implements an attack where an intermediary finds all potential sources and destinations of a transaction that it is a part of.

*execute.py* - Runs a sample experiment with a set of adversaries on transactions between random pairs of sources and destinations.

*stats.py* - Generates statistics and plots from the anonymity sets.

## Running an experiment

- Choose the set of adversaries using the centrality data from data/Centrality as your guide.
- Specify the number of transactions you want to simulate in a run.
- Specify the mechanism for choosing source destination pairs for each transaction.
- The details of the transactions and the results of each attack instance is copied to *transactions.csv* and *results.csv* respectively.

## Results

The following results can be obtained from *results.py*:
- A figure plotting the sizes of each source and destination anonymity set.
- Number of transactions for which either source or the destination anonymity set was a singular set.
- The number of transactions attacked, the average size of source and destination anonymity sets returned by each adversary.
- Correlation of the number of hops from the adversary to source/destination to the size of source/destination anonymity set.

