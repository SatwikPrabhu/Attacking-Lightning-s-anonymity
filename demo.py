import networkx as nx
import json
import random as rn
from math import inf
from queue import  PriorityQueue
import nested_dict as nd

LND_RISK_FACTOR = 0.000000015

def lnd_cost_fun(G, amount, u, v):
    fee = G.edges[v,u]['BaseFee'] + amount * G.edges[v, u]['FeeRate']
    alt = (amount+fee) * G.edges[v, u]["Delay"] * LND_RISK_FACTOR + fee
    return alt

def Dijkstra(G,source,target,amt,cost_function):
    paths = {}
    dist = {}
    delay = {}
    amount =  {}
    # prob = {}
    for node in G.nodes():
        amount[node] = -1
        delay[node] = -1
        dist[node] = inf
    visited = set()
    pq = PriorityQueue()
    dist[target] = 0
    delay[target] = 0
    paths[target] = [target]
    amount[target] = amt
    pq.put((dist[target],target))
    while 0!=pq.qsize():
        curr_cost,curr = pq.get()
        if curr == source:
            return paths[curr],delay[curr],amount[curr],dist[curr]
        if curr_cost > dist[curr]:
            continue
        visited.add(curr)
        for [v,curr] in G.in_edges(curr):
            if v == source and G.edges[v,curr]["Balance"]>=amount[curr]:
                cost = dist[curr] + amount[curr]*G.edges[v,curr]["Delay"]*LND_RISK_FACTOR
                if cost < dist[v]:
                    dist[v] = cost
                    paths[v] = [v] + paths[curr]
                    delay[v] = G.edges[v, curr]["Delay"] + delay[curr]
                    amount[v] = amount[curr]
                    pq.put((dist[v],v))
            if(G.edges[v, curr]["Balance"] + G.edges[curr, v]["Balance"] >= amount[curr]) and v not in visited and v!=source:
                cost = dist[curr] + cost_function(G,amount[curr],curr,v)
                if cost < dist[v]:
                    dist[v] = cost
                    paths[v] = [v] + paths[curr]
                    delay[v] = G.edges[v,curr]["Delay"] + delay[curr]
                    amount[v] = amount[curr] + G.edges[v,curr]["BaseFee"] + amount[curr]*G.edges[v,curr]["FeeRate"]
                    pq.put((dist[v],v))
    return [],-1,-1,-1

def dest_reveal_new(G,adversary,delay,amount,pre,next):
    T = nd.nested_dict()
    flag1 = True
    anon_sets = nd.nested_dict()
    level = 0
    T[0]["nodes"] = [next]
    T[0]["delays"] = [delay]
    T[0]["previous"] = [-1]
    T[0]["visited"] = [[pre,adversary,next]]
    T[0]["amounts"] = [amount]
    flag = True

    while(flag):
        level+=1
        if(level == 4):
            flag1 = False
            break
        t1 = T[level - 1]["nodes"]
        d1 = T[level - 1]["delays"]
        v1 = T[level - 1]["visited"]
        a1 = T[level - 1]["amounts"]
        t2 = []
        d2 = []
        p2 = []
        v2 = [[]]
        a2 = []
        for i in range(0,len(t1)):
            u = t1[i]
            for [u,v] in G.out_edges(u):
                if(v!=pre and v!=adversary  and v!=next and v not in v1[i] and (d1[i] - G.edges[u,v]["Delay"])>=0 and (G.edges[u,v]["Balance"]+G.edges[v,u]["Balance"])>=((a1[i] - G.edges[u, v]["BaseFee"]) / (1 + G.edges[u, v]["FeeRate"]))):
                    t2.append(v)
                    d2.append(d1[i] - G.edges[u,v]["Delay"])
                    p2.append(i)
                    v2.append(v1[i]+[v])
                    a2.append(((a1[i] - G.edges[u, v]["BaseFee"]) / (1 + G.edges[u, v]["FeeRate"])))
        T[level]["nodes"] = t2
        T[level]["delays"] = d2
        T[level]["previous"] = p2
        T[level]["visited"] = v2
        T[level]["amounts"] = a2
        if(len(t2) == 0):
            flag = False
    level = level - 1
    while(level>=0):
        t = T[level]["nodes"]
        d = T[level]["delays"]
        p = T[level]["previous"]
        a = T[level]["amounts"]
        v = T[level]["visited"]
        for i in range(0, len(t)):
            if(d[i] == 0):
                path = []
                level1 = level
                path.append(T[level1]["nodes"][i])
                loc = T[level1]["previous"][i]
                while (level1 > 0):
                    level1 = level1 - 1
                    path.append(T[level1]["nodes"][loc])
                    loc = T[level1]["previous"][loc]
                path.reverse()
                path = [pre,adversary]+path
                if (len(path) == len(set(path))):
                    amt = a[i]
                    pot = path[len(path) - 1]
                    sources = deanonymize(G,pot,path,amt,lnd_cost_fun)
                    if sources != None:
                        anon_sets[pot] = list(sources)
        level = level - 1
    return anon_sets,flag1

def deanonymize(G,target,path,amt,cost_function):
    pq = PriorityQueue()
    delays = {}
    costs = {}
    paths = nd.nested_dict()
    paths1 = nd.nested_dict()
    dists = {}
    visited = set()
    previous = {}
    done = {}
    prob = {}
    sources = []
    pre = path[0]
    adv = path[1]
    nxt = path[2]
    for node in G.nodes():
        previous[node] = -1
        delays[node] = -1
        costs[node] = inf
        paths[node] = []
        dists[node] = inf
        done[node] = 0
        paths1[node] = []
        prob[node] = 1
    dists[target] = 0
    paths[target] = [target]
    costs[target] = amt
    delays[target] = 0
    pq.put((dists[target],target))
    flag1 = 0
    flag2 = 0
    while(0!=pq.qsize()):
        curr_cost,curr = pq.get()
        if curr_cost > dists[curr]:
            continue
        visited.add(curr)
        for [v,curr] in G.in_edges(curr):
            if (G.edges[v, curr]["Balance"] + G.edges[curr, v]["Balance"] >= costs[curr]) and v not in visited:
                if done[v] == 0:
                    paths1[v] = [v]+paths[curr]
                    done[v] = 1
                cost = dists[curr]+ cost_function(G,costs[curr],curr,v)
                if cost < dists[v]:
                    paths[v] = [v]+paths[curr]
                    dists[v] = cost
                    delays[v] = delays[curr] + G.edges[v,curr]["Delay"]
                    costs[v] = costs[curr] + G.edges[v, curr]["BaseFee"] + costs[curr] * G.edges[v, curr]["FeeRate"]
                    pq.put((dists[v],v))
        if(curr in path[1:]):
            ind = path.index(curr)
            if(paths[curr]!=path[ind:]):
                return None
            if curr == adv:
                flag1 = 1
        if(curr == pre):
            if paths[pre] != path:
                return [pre]
            else:
                sources.append(pre)
            flag2 = 1
        if flag1 == 1 and flag2 == 1:
            if pre in paths[curr]:
                for [v,curr] in G.in_edges(curr):
                        if v not in paths[curr]:
                            sources.append(v)
    sources = set(sources)
    return sources

def route(G,path,delay,amt,ads,amt1,file):
    G1 = G.copy()
    cost = amt
    comp_attack = []
    anon_sets =[]
    attacked = 0
    G.edges[path[0],path[1]]["Balance"] -= amt
    G.edges[path[1],path[0]]["Locked"] = amt
    delay = delay - G.edges[path[0],path[1]]["Delay"]
    i = 1
    if len(path) == 2:
        G.edges[path[1],path[0]]["Balance"] += G.edges[path[1],path[0]]["Locked"]
        G.edges[path[1], path[0]]["Locked"] = 0
        transaction = {"sender": path[0], "recipient": path[1], "path" : path, "delay": delay, "amount":amt1,
                       "Cost": cost, "attacked":0,
                    "success":True,"anon_sets":anon_sets,"comp_attack":comp_attack}
        transactions.append(transaction)
        return True
    while(i < len(path)-1):
        amt = (amt - G.edges[path[i], path[i+1]]["BaseFee"]) / (1 + G.edges[path[i], path[i+1]]["FeeRate"])
        if path[i] in ads:
            attacked+=1
            dests = []
            delay1 = delay - G.edges[path[i],path[i+1]]["Delay"]
            B,flag = dest_reveal_new(G1,path[i],delay1,amt,path[i-1],path[i+1])
            for j in B:
                dest = {j:B[j]}
                dests.append(dest)
            if flag == True:
                comp_attack.append(1)
            else:
                comp_attack.append(0)
            anon_set = {path[i]:dests}
            anon_sets.append(anon_set)
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
                    if j==0:
                        transaction = {"sender": path[0], "recipient": path[len(path)-1], "path": path, "delay": delay,
                                       "amount": amt1, "Cost": cost, "attacked": attacked,
                                       "success": True, "anon_sets": anon_sets, "comp_attack": comp_attack}
                        transactions.append(transaction)
                    j = j-1
                return True
            delay = delay - G.edges[path[i],path[i+1]]["Delay"]
            i += 1
        else:
            j = i - 1
            while j >= 0:
                G.edges[path[j],path[j+1]]["Balance"] += G.edges[path[j+1],path[j]]["Locked"]
                G.edges[path[j + 1], path[j]]["Locked"] = 0
                if j==0:
                    transaction = {"sender": path[0], "recipient": path[len(path)-1], "path": path, "delay": delay,
                                   "amount": amt1, "Cost": cost, "attacked": attacked,
                                   "success": False, "anon_sets": anon_sets, "comp_attack": comp_attack}
                    transactions.append(transaction)
                j = j-1
            return False

G = nx.barabasi_albert_graph(100,2,65)
G = nx.DiGraph(G)

for [u,v] in G.edges():
    G.edges[u,v]["Delay"] = 10 * rn.randint(1,10)
    G.edges[u,v]["BaseFee"] = 0.1 * rn.randint(1,10)
    G.edges[u,v]["FeeRate"] = 0.0001 * rn.randint(1,10)
    G.edges[u,v]["Balance"] = rn.randint(100,10000)

transactions = []

B = nx.betweenness_centrality(G)

ads = []
for i in range(0,10):
    node = -1
    max = -1
    for u in G.nodes():
        if B[u] >= max and u not in ads:
            max = B[u]
            node = u
    if node not in ads:
        ads.append(node)

print("Adversaries:",ads)

i=0
file = "results.json"
while (i < 100):
    u = -1
    v = -1
    while (u == v or (u not in G.nodes()) or (v not in G.nodes())):
        u = rn.randint(0, 99)
        v = rn.randint(0, 99)
    amt = 0
    if (i % 3 == 0):
        amt = rn.randint(1, 10)
    elif (i % 3 == 1):
        amt = rn.randint(10, 100)
    elif (i % 3 == 2):
        amt = rn.randint(100, 1000)
    path, delay, amount, dist = Dijkstra(G, u, v, amt, lnd_cost_fun)
    if (len(path) > 0):
        T = route(G, path, delay, amount, ads, amt, file)
    if len(path) > 2:
        print(i,path, "done")
        i += 1
with open(file,'r') as json_file:
    data = json.load(json_file)
data.append(transactions)
with open(file,'w') as json_file:
    json.dump(data,json_file,indent=1)
