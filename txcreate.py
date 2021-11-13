import numpy as np
import networkx as nx
import random as rn
import populate_graph as pg


n = 16000
txs = np.zeros((n,3), dtype=int)

G = nx.DiGraph()
G,m = pg.populate_nodes(G)
G,m1=pg.populate_channels(G,m,645320)
G = pg.populate_policies(G,m1)


G1 = nx.DiGraph()
for [u,v] in G.edges():
    if(G.edges[u,v]["marked"]==1 and G.edges[v,u]["marked"]==1):
        if (u not in G1.nodes()):
            G1.add_node(u)
            G1.nodes[u]["name"] = G.nodes[u]["name"]
            G1.nodes[u]["pubadd"] = G.nodes[u]["pubadd"]
            G1.nodes[u]["Tech"] = G.nodes[u]["Tech"]
        if (v not in G1.nodes()):
            G1.add_node(v)
            G1.nodes[v]["name"] = G.nodes[v]["name"]
            G1.nodes[v]["pubadd"] = G.nodes[v]["pubadd"]
            G1.nodes[v]["Tech"] = G.nodes[v]["Tech"]
        G1.add_edge(u,v)
        G1.edges[u,v]["Balance"] = G.edges[u,v]["Balance"]
        G1.edges[u, v]["Age"] = G.edges[u, v]["Age"]
        G1.edges[u, v]["BaseFee"] = G.edges[u, v]["BaseFee"]
        G1.edges[u, v]["FeeRate"] = G.edges[u, v]["FeeRate"]
        G1.edges[u, v]["Delay"] = G.edges[u, v]["Delay"]
        G1.edges[u, v]["id"] = G.edges[u, v]["id"]
        
print("node count: ", len(G1.nodes()))
i = 0
while(i<n):
    u = -1
    v = -1
    # We go for random source/destination pairs. This can be changed to having a biased choice as well
    while (u == v or (u not in G1.nodes()) or (v not in G1.nodes())):
        u = rn.randint(0, len(G1.nodes()) - 1)
        v = rn.randint(0, len(G1.nodes()) - 1)
    # Try to get an exponential distribution for transaction amounts. This can be changed as well.
    j = rn.randint(0,4)
    if (j % 5 == 1):
        amt = rn.randint(1, 10)
    elif (j % 5 == 2):
        amt = rn.randint(10, 100)
    elif (j % 5 == 3):
        amt = rn.randint(100, 1000)
    elif (j % 5 == 4):
        amt = rn.randint(1000, 10000)
    else:
        amt = rn.randint(10000, 100000)
    txs[i,0] = u
    txs[i,1] = v
    txs[i,2] = amt
    i += 1

np.savetxt("txs.txt", txs, fmt='%d')
print("done")