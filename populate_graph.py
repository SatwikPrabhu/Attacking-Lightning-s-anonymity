
import json
import networkx as nx
import random as rn
import csv
import string
import re
import ast

# read node information and initialize the graph with the nodes
def populate_nodes(G, tech=-1):
    node_list = []
    map = dict()
    with open("nodes1.csv", 'r') as csv_file:
        csvreader = csv.reader(csv_file)
        line = 0
        for row in csvreader:
            if (line != 0):
                G.add_node(line - 1)
                G.nodes[line - 1]["name"] = row[2]
                G.nodes[line - 1]["pubadd"] = row[1]
                # G.nodes[line - 1]["Tech"] = rn.randint(0, 2)
                if (tech == -1):
                    if row[4] == 'c-lightning':
                        G.nodes[line - 1]["Tech"] = 1
                    elif row[4] == 'lnd':
                        G.nodes[line - 1]["Tech"] = 0
                    elif row[4] == 'eclair':
                        G.nodes[line - 1]["Tech"] = 2
                    else:
                        G.nodes[line - 1]["Tech"] = -1
                elif tech == 0:
                    G.nodes[line - 1]["Tech"] = 0
                elif tech == 1:
                    G.nodes[line - 1]["Tech"] = 1
                elif tech == 2:
                    G.nodes[line - 1]["Tech"] = 2
                map[row[1]] = line - 1
            line += 1
    return G, map

# Add channel information with age and total capacities
def populate_channels(G, map, cbr):
    map1 = dict()
    with open("channels.csv", 'r') as csv_file:
        csvreader = csv.reader(csv_file)
        line = 0
        for row in csvreader:
            if (line != 0):
                # print(line)
                id = row[1]
                nodes = ast.literal_eval(row[3])
                u = map[nodes[0]]
                v = map[nodes[1]]
                G.add_edge(u, v)
                G.add_edge(v, u)
                opens = row[6]
                # print(opens.split()[7])
                s = opens.split()[7]
                s = re.findall("\d+", s)
                blk = int(s[0])

                G.edges[u, v]["Age"] = blk
                G.edges[u, v]["id"] = id
                G.edges[v, u]["Age"] = blk
                G.edges[v, u]["id"] = id
                # We randomly distribute the capacity in both directions initially.
                x = rn.uniform(0,int(row[2]))
                G.edges[u, v]["Balance"] = x
                G.edges[v, u]["Balance"] = int(row[2]) - x
                map1[id] = [u, v]
                G.edges[u, v]["marked"] = 0
                G.edges[v, u]["marked"] = 0
                #Last failure is used in lnd for calculating edge probability. We ignore this aspect for now
                # G.edges[u, v]["LastFailure"] = 25
                # G.edges[v, u]["LastFailure"] = 25

            line += 1
    return G, map1

# add fee and delay policies
def populate_policies(G, map):
    with open("policies.csv", 'r') as csv_file:
        csvreader = csv.reader(csv_file)
        line = 0
        channels = []
        for row in csvreader:
            if (line != 0):
                id = row[1]
                if (id in map):
                    # if(id not in channels):
                    #     channels.append(id)
                    nodes = map[id]
                    u = nodes[0]
                    v = nodes[1]
                    # print(int(row[2]))
                    if (int(row[2]) == 0):
                        G.edges[u, v]["BaseFee"] = int(row[3]) / 1000
                        G.edges[u, v]["FeeRate"] = int(row[4]) / 1000000
                        G.edges[u, v]["Delay"] = int(row[5])
                        # print(G.edges[u,v]["id"],G.edges[u,v]["Delay"])
                        G.edges[u, v]["marked"] = 1
                    elif (int(row[2]) == 1):
                        G.edges[v, u]["BaseFee"] = int(row[3]) / 1000
                        G.edges[v, u]["FeeRate"] = int(row[4]) / 1000000
                        G.edges[v, u]["Delay"] = int(row[5])
                        # print(G.edges[v, u]["id"],G.edges[v,u]["Delay"])
                        G.edges[v, u]["marked"] = 1

            line += 1
    # print(len(channels))
    return G
