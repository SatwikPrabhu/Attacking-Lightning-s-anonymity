import pathFind as pf
import populate_graph as pg
import attack_blind as at
import networkx as nx
import sys
import random as rn
import nested_dict as nd
import json
from networkx.readwrite import json_graph

file = []

transactions = []

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


def route(G,path,delay,amt,ads,amt1,file,tech):
    cost = amt
    comp_attack = []
    anon_sets =[]
    attacked = 0
    G1 = G.copy()
    #print(path[0])
    G.edges[path[0],path[1]]["Balance"] -= amt
    G.edges[path[1],path[0]]["Locked"] = amt
    delay = delay - G.edges[path[0],path[1]]["Delay"]
    i = 1
    if len(path) == 2:
        G.edges[path[1],path[0]]["Balance"] += G.edges[path[1],path[0]]["Locked"]
        G.edges[path[1], path[0]]["Locked"] = 0
        # with open(file) as json_file:
        #     data = json.load(json_file)
        #     temp = data
        transaction = {"sender": path[0], "recipient": path[1], "path" : path, "delay": delay, "amount":amt1, "Cost": cost, "tech":tech,
                       "attacked":0,
                    "success":True,"anon_sets":anon_sets,"comp_attack":comp_attack}
        transactions.append(transaction)
        # with open(file,'w') as json_file:
        #     json.dump(temp,json_file,indent=4)
        return True
    while(i < len(path)-1):
        #print(path[i])
        amt = (amt - G.edges[path[i], path[i+1]]["BaseFee"]) / (1 + G.edges[path[i], path[i+1]]["FeeRate"])
        if path[i] in ads:
            attacked+=1
            dests = []
            delay1 = delay - G.edges[path[i],path[i+1]]["Delay"]
            #print(delay1)
            B,flag = at.dest_reveal_new(G1,path[i],delay1,amt,path[i-1],path[i+1])
            for j in B:
                dest = {j:B[j]}
                dests.append(dest)
            if flag == True:
                comp_attack.append(1)
            else:
                comp_attack.append(0)
            anon_set = {path[i]:dests}
            anon_sets.append(anon_set)
            # with open(file,'a') as csv_file:
            #     csvwriter = csv.writer(csv_file)
            #     for j in B:
            #         csvwriter.writerow([ind,path[i],j,B[j],flag])
        if(G.edges[path[i],path[i+1]]["Balance"] >= amt):
            G.edges[path[i], path[i+1]]["Balance"] -= amt
            G.edges[path[i+1], path[i]]["Locked"] = amt
            if i == len(path) - 2:
                G.edges[path[i+1],path[i]]["Balance"] += G.edges[path[i+1], path[i]]["Locked"]
                G.edges[path[i+1], path[i]]["Locked"] = 0
                j = i - 1
                while j >= 0:
                    G.edges[path[j + 1], path[j]]["Balance"] += G.edges[path[j + 1], path[j]]["Locked"]
                    G.edges[path[j + 1], path[j]]["Locked"] = 0

                # with open(file) as json_file:
                #     data = json.load(json_file)
                #     temp = data
                    if j == 0:
                        transaction = {"sender": path[0], "recipient": path[len(path)-1], "path": path, "delay": delay,
                                       "amount": amt1, "Cost": cost,"tech":tech, "attacked": attacked,
                                       "success": True, "anon_sets": anon_sets, "comp_attack": comp_attack}
                        transactions.append(transaction)
                    j=j-1
                # with open(file, 'w') as json_file:
                #     json.dump(temp, json_file,indent=4)
                return True
            delay = delay - G.edges[path[i],path[i+1]]["Delay"]
            i += 1
        else:
            # G.edges[path[i],path[i+1]]["LastFailure"] = 0
            j = i - 1
            while j >= 0:
                G.edges[path[j],path[j+1]]["Balance"] += G.edges[path[j+1],path[j]]["Locked"]
                G.edges[path[j + 1], path[j]]["Locked"] = 0
            # with open(file) as json_file:
            #     data = json.load(json_file)
            #     temp = data
                if j == 0:
                    transaction = {"sender": path[0], "recipient": path[len(path)-1], "path": path, "delay": delay,
                                   "amount": amt1, "Cost": cost,"tech":tech, "attacked": attacked,
                                   "success": False, "anon_sets": anon_sets, "comp_attack": comp_attack}
                    transactions.append(transaction)
                j = j-1
            # with open(file, 'w') as json_file:
            #     json.dump(temp, json_file,indent = 4)
            return False


if __name__ == "__main__":
    a = int(sys.argv[1])
    ads = [2634,5422, 8075, 5347, 1083, 5093,4326, 4126, 2836, 5361, 10572,5389, 3599, 9819, 4828, 3474, 8808, 93, 9530, 9515, 2163]
    i = 0
    while (i <= 10):
        # with open("graph.json",'r') as json_file:
        #     graph_json = json.load(json_file)
        # G = json_graph.node_link_graph(graph_json)
        u = -1
        v = -1
        while (u == v or (u not in G1.nodes()) or (v not in G1.nodes())):
            u = rn.randint(0, 11197)
            v = rn.randint(0, 11197)
            # if(u!=v):
            # i+=1
        if (i % 5 == 1):
            amt = rn.randint(1, 10)
        elif (i % 5 == 2):
            amt = rn.randint(10, 100)
        elif (i % 5 == 3):
            amt = rn.randint(100, 1000)
        elif (i % 5 == 4):
            amt = rn.randint(1000, 10000)
        else:
            amt = rn.randint(10000, 100000)
        #print(u, v, amt,G1.nodes[u]["Tech"])
        if (G1.nodes[u]["Tech"] == 0):
            path, delay, amount, dist = pf.Dijkstra(G1, u, v, amt, pf.lnd_cost_fun)
        elif (G1.nodes[u]["Tech"] == 1):
            fuzz = rn.uniform(-1, 1)
            path, delay, amount, dist = pf.Dijkstra(G1, u, v, amt, pf.c_cost_fun(fuzz))
        else:
            paths = pf.Dijkstra_general(G1, u, v, amt, pf.eclair_cost_fun)
            if (paths[0] == []):
                continue
            if len(paths[0]) == 2:
                path = paths[0]
                r = 0
            else:
                r = rn.randint(0, 2)
                path = paths[r]
            delay = 0
            amount = amt
            #print(path, r)
            if (len(path) > 2):
                for m in range(len(path) - 2, 0, -1):
                    delay += G1.edges[path[m], path[m + 1]]["Delay"]
                    amount += G1.edges[path[m], path[m + 1]]["BaseFee"] + amount * G1.edges[path[m], path[m + 1]][
                        "FeeRate"]
                delay += G1.edges[path[0], path[1]]["Delay"]
        if(len(path)>0):
            T = route(G1,path,delay,amount,ads,amt,file[a],G1.nodes[u]["Tech"])
        # graph_json = json_graph.node_link_data(G)
        # with open("graph.json", 'w') as json_file:
        #     json.dump(graph_json,json_file)
        if len(path)>2:
            #print(i,"done")
            i+=1
    with open(file[a],'r') as json_file:
        data = json.load(json_file)
    data.append(transactions)
    with open(file[a],'w') as json_file:
        json.dump(data,json_file,indent=1)
