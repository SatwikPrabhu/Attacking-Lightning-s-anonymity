import csv
import ast
import nested_dict as nd
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns
import networkx as nx
import populate_graph as pg

file = ["results8/results_blind1.json",
        "results8/results_blind2.json",
        "results8/results_blind3.json","results8/results_blind4.json","results8/results_blind5.json"]

all_results = []
for i in range(0,5):
    with open(file[i],'r') as json_file:
        results_json = json.load(json_file)
    all_results.append(results_json)

G = nx.DiGraph()
G,m = pg.populate_nodes(G)
G,m1=pg.populate_channels(G,m,645320)
G = pg.populate_policies(G,m1)

G1 = nx.DiGraph()
# for node in G.nodes():
#     G1.add_node(node)
#     G1.nodes[node]["name"] = G.nodes[node]["name"]
#     G1.nodes[node]["pubadd"] = G.nodes[node]["pubadd"]
#     G1.nodes[node]["Tech"] = G.nodes[node]["Tech"]
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

nodes = []
for u in G1.nodes():
    nodes.append(u)
nodes = set(nodes)
path = []
dest_count_incomp = []
dest_count_comp = []
source_count_incomp = []
source_count_comp = []
ads = [2634, 5422, 8075, 5347, 1083, 5093, 4326, 4126, 2836, 5361, 10572, 5389, 3599, 9819, 4828, 3474, 8808, 93, 9530,
       9515, 2163]
ad_attacks = {}
for ad in ads:
    ad_attacks[ad] = 0
for i in all_results:
    for j in i:
        for k in j:
            if k["path"]!=path:
                path = k["path"]
                if k["attacked"]>0:
                    anon_sets = k["anon_sets"]
                    if len(anon_sets)>1:
                        print(k["comp_attack"])
                        sources_comp = nodes
                        sources_incomp = nodes
                        dests_comp = nodes
                        dests_incomp = nodes
                        att = -1
                        for ad in anon_sets:
                            att += 1
                            for adv in ad:
                                s_comp = []
                                s_incomp = []
                                d_comp = []
                                d_incomp = []
                                ind = k["path"].index(int(adv))
                                dist_dest = len(k["path"])-1-ind
                                dist_source = ind
                                if k["comp_attack"][att] == 1:
                                    for dest in ad[adv]:
                                        for rec in dest:
                                            for tech in dest[rec]:
                                                d_comp.append(int(rec))
                                                for s in dest[rec][tech]:
                                                    if int(tech) == G1.nodes[s]["Tech"]:
                                                        s_comp.append(s)
                                    d = set(d_comp)
                                    so = set(s_comp)
                                    dests_comp = dests_comp.intersection(d)
                                    sources_comp = sources_comp.intersection(so)
                                for dest in ad[adv]:
                                    for rec in dest:
                                        for tech in dest[rec]:
                                            d_incomp.append(int(rec))
                                            for s in dest[rec][tech]:
                                                if int(tech) == G1.nodes[s]["Tech"]:
                                                    s_incomp.append(s)
                                d = set(d_incomp)
                                so = set(s_incomp)
                                dests_incomp = dests_incomp.intersection(d)
                                sources_incomp = sources_incomp.intersection(so)
                        dest_count_comp.append(len(dests_comp))
                        dest_count_incomp.append(len(dests_incomp))
                        source_count_comp.append(len(sources_comp))
                        source_count_incomp.append(len(sources_incomp))

new_dest_count_comp = []
new_source_count_comp = []
new_dest_count_incomp = []
new_source_count_incomp = []

for i in range(0,len(dest_count_comp)):
    if dest_count_comp[i] != 0 and dest_count_comp[i]!=4791:
        new_dest_count_comp.append(dest_count_comp[i])

for i in range(0,len(source_count_comp)):
    if source_count_comp[i] != 0 and dest_count_comp[i]!=4791:
        new_source_count_comp.append(source_count_comp[i])

for i in range(0,len(dest_count_incomp)):
    if dest_count_incomp[i] != 0:
        new_dest_count_incomp.append(dest_count_incomp[i])

for i in range(0,len(source_count_incomp)):
    if source_count_incomp[i] != 0:
        new_source_count_incomp.append(source_count_incomp[i])

print(new_dest_count_comp)
print(new_dest_count_incomp)
print(new_source_count_comp)
print(new_source_count_incomp)

plot1 = sns.ecdfplot(data = new_dest_count_comp,legend='Adversaries who completed Phase I',marker = '|',linewidth = 1.5, linestyle = ':')
plot2 = sns.ecdfplot(data = new_dest_count_incomp,legend='All adversaries',marker = '|',linewidth = 1.5, linestyle = ':')
plot1.set(xscale='log')
plot2.set(xscale='log')
plt.legend(('Adversaries who completed Phase I','All adversaries'),scatterpoints=1,loc='lower right',ncol=1,fontsize=16)
plt.xlabel("Size of anonymity set")
plt.ylabel("CDF")
plt.show()

plot1 = sns.ecdfplot(data = new_source_count_comp,legend='Adversaries who completed Phase I',marker = '|',linewidth = 1.5, linestyle = ':')
plot2 = sns.ecdfplot(data = new_source_count_incomp,legend='All adversaries',marker = '|',linewidth = 1.5, linestyle = ':')
plot1.set(xscale='log')
plot2.set(xscale='log')
plt.legend(('Adversaries who completed Phase I','All adversaries'),scatterpoints=1,loc='lower right',ncol=1,fontsize=16)
plt.xlabel("Size of anonymity set")
plt.ylabel("CDF")
plt.show()


