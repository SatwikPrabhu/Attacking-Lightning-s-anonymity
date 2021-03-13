# Evaluation of Lightning's Anonymity
An attack that enables an intermediary to break the anonymity of the source and destination of a trannsaction in the Lightning network. 

This includes a simulator to simulate transactions using LND, c-Lightning and Eclair. We modify Eclair to use a generalized version of Dijkstra's algorithm instead of using Yen's algorithm. We do not have code to simulate concurrent payments yet. 

The experiment is run on a snapshot of the Lightning Network obtained from https://ln.bigsun.xyz. The set of adversaries is a mixture of nodes with high centrality, low centrality and random nodes. The snapshot as well as the centralities of all nodes are found in data/Snapshot and data/Centrality respectively.

