import networkx as nx
import nested_dict as nd
import random as rn
import matplotlib.pyplot as plt
import time
import requests
import json
from queue import  PriorityQueue
import math
from math import inf
import heapq
from math import isinf

CBR = 648601    # TODO: UPDATE CURRENT BLOCK BEFORE RUNNING!

# lnd
LND_RISK_FACTOR = 0.000000015
A_PRIORI_PROB = 0.6

# c_lightning
C_RISK_FACTOR = 10
RISK_BIAS = 1
DEFAULT_FUZZ = 0.05

# eclair
MIN_DELAY = 9
MAX_DELAY = 2016
MIN_CAP = 1
MAX_CAP = 100000000
MIN_AGE = 505149
MAX_AGE = CBR
DELAY_RATIO = 0.15
CAPACITY_RATIO = 0.5
AGE_RATIO = 0.35


def normalize(val, min, max):
    if val <= min:
        return 0.00001
    if val > max:
        return 0.99999
    return (val - min) / (max - min)

def edge_prob(t):
    if t<1:
        return 0
    if t>=25:
        return 0.6
    return (A_PRIORI_PROB * (1 - 1 / (2 ** t)))

def prob_bias(dist,prob):
    if prob < 0.00001:
        return inf
    return dist+(100/prob)

def shortest_cost_fun(G,amount,prev,u,v,direct_conn=False):
    return 1

def least_delay_cost_fun(G,amount,prev,u,v,direct_conn=False):
    return G.edges[v,u]["Delay"]

def cheapest_cost_fun(G,amount,prev,u,v,direct_conn=False):
    if(direct_conn):
        return 0
    fee = G.edges[u, prev]['BaseFee'] + amount[u] * G.edges[u, prev]['FeeRate']
    return fee


def lnd_cost_fun(G, amount, u, v):
    # if direct_conn:
    #     return amount[v] * G.edges[v, u]["Delay"] * LND_RISK_FACTOR
    fee = G.edges[v,u]['BaseFee'] + amount * G.edges[v, u]['FeeRate']
    alt = (amount+fee) * G.edges[v, u]["Delay"] * LND_RISK_FACTOR \
          + fee

    # t = G.edges[v, u]["LastFailure"]
    # edge_proba = edge_prob(t)
    # edge_proba *= prob
    # alt = prob_bias(alt,edge_proba)
    return alt

def c_cost_fun(fuzzfactor):
    def fun(G, amount, u, v):
        # if direct_conn:
        #     return amount[v] * G.edges[v, u]["Delay"] * C_RISK_FACTOR + RISK_BIAS
        scale = 1 + DEFAULT_FUZZ * fuzzfactor
        fee = scale * (G.edges[v, u]['BaseFee'] + amount * G.edges[v, u]['FeeRate'])
        alt = ((amount + fee) * G.edges[v, u]["Delay"] * C_RISK_FACTOR + RISK_BIAS)
        return alt
    return fun


def eclair_cost_fun(G, amount, u, v):
    # if direct_conn:
    #     return 0
    fee = G.edges[v,u]['BaseFee'] + amount * G.edges[v,u]['FeeRate']
    ndelay = normalize(G.edges[v, u]["Delay"], MIN_DELAY, MAX_DELAY)
    ncapacity = 1 - (normalize((G.edges[v, u]["Balance"] + G.edges[u, v]["Balance"]), MIN_CAP, MAX_CAP))
    nage = normalize(CBR - G.edges[v, u]["Age"], MIN_AGE, MAX_AGE)
    alt = fee * (ndelay * DELAY_RATIO + ncapacity * CAPACITY_RATIO + nage * AGE_RATIO)
    return alt

def fee_cost_fun(G, amount, prev, u, v, direct_conn=False):
    if direct_conn:
        return 0
    fee = G.edges[u, prev]['BaseFee'] + amount[u] * G.edges[u, prev]['FeeRate']
    return fee


def delay_cost_fun(G, amount, prev, u, v, direct_conn=False):
    return G.edges[v, u]["Delay"]


def build_path(u, previous):
    path = []
    while previous[u] != -1:
        path.append(u)
        #print(u)
        u = previous[u]
        #print(u)
    path.append(u)
    return path




def calc_params(G, path, amt):
    cost = amt
    #print(path)
    delay = 0
    dist = 0
    for i in range(len(path)-2,0,-1):
        #print(path[i],path[i+1])
        #print(G.edges[path[i],path[i+1]]["FeeRate"])
        fee = cost * G.edges[path[i], path[i+1]]["FeeRate"] + G.edges[path[i], path[i+1]]["BaseFee"]

        ndelay = normalize(G.edges[path[i], path[i+1]]["Delay"], MIN_DELAY, MAX_DELAY)
        ncapacity = normalize((G.edges[path[i], path[i+1]]["Balance"] + G.edges[path[i], path[i+1]]["Balance"]), MIN_CAP, MAX_CAP)
        nage = normalize(CBR - G.edges[path[i], path[i+1]]["Age"], MIN_AGE, MAX_AGE)

        dist += fee * (ndelay * DELAY_RATIO + ncapacity * CAPACITY_RATIO + nage * AGE_RATIO)
        delay += G.edges[path[i],path[i+1]]["Delay"]
        cost += fee
    return dist

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
        # prob[node] = 1
    prev = {}
    visited = set()
    pq = PriorityQueue()
    dist[target] = 0
    delay[target] = 0
    #print(dist[target])
    paths[target] = [target]
    #print(dist[target])
    amount[target] = amt
    #print(dist[target],amount[target])
    pq.put((dist[target],target))
    #print(dist[target])
    #print(pq)
    while 0!=pq.qsize():
        curr_cost,curr = pq.get()
        #print(curr)
        #print(pq)
        if curr == source:
            return paths[curr],delay[curr],amount[curr],dist[curr]
        if curr_cost > dist[curr]:
            continue
        visited.add(curr)
        #print(curr)
        for [v,curr] in G.in_edges(curr):
            if v == source and G.edges[v,curr]["Balance"]>=amount[curr]:
                #print(curr)
                if(G.nodes[source]["Tech"] == 0):
                    cost = dist[curr] + amount[curr]*G.edges[v,curr]["Delay"]*LND_RISK_FACTOR
                elif(G.nodes[source]["Tech"] == 1):
                    cost = dist[curr] + (amount[curr] * G.edges[v, curr]["Delay"] * C_RISK_FACTOR + RISK_BIAS)
                else:
                    cost = dist[curr]
                if cost < dist[v]:
                    dist[v] = cost
                    paths[v] = [v] + paths[curr]
                    delay[v] = G.edges[v, curr]["Delay"] + delay[curr]
                    amount[v] = amount[curr]
                    pq.put((dist[v],v))
                # return [v]+paths[curr],delay[curr]+G.edges[v,curr]["Delay"],amount[curr],dist[curr]
            if(G.edges[v, curr]["Balance"] + G.edges[curr, v]["Balance"] >= amount[curr]) and v not in visited:
                cost = dist[curr] + cost_function(G,amount[curr],curr,v)
                if cost < dist[v]:
                    dist[v] = cost
                    paths[v] = [v] + paths[curr]
                    delay[v] = G.edges[v,curr]["Delay"] + delay[curr]
                    amount[v] = amount[curr] + G.edges[v,curr]["BaseFee"] + amount[curr]*G.edges[v,curr]["FeeRate"]
                    # t = G.edges[v,curr]["LastFailure"]
                    # edge = edge_prob(t)
                    # prob[v] = edge*prob[curr]
                    pq.put((dist[v],v))
    return [],-1,-1,-1

def Dijkstra_all_paths(G,target,amt,cost_function):
    paths = {}
    dist = {}
    delay = {}
    amount = {}
    done = {}
    prev = {}
    # prob = {}
    visited = set()
    for node in G.nodes():
        amount[node] = -1
        delay[node] = -1
        dist[node] = inf
        done[node] = 0
        prev[node] = -1
        # prob[node] = 1
    if cost_function == lnd_cost_fun(prob=1):
        tech = 0
    elif cost_function ==eclair_cost_fun:
        tech = 2
    else:
        tech = 1
    visited = set()
    pq = PriorityQueue()
    dist[target] = 0
    delay[target] = 0
    # print(dist[target])
    paths[target] = [target]
    # print(dist[target])
    amount[target] = amt
    done[target] = 1
    # print(dist[target],amount[target])
    pq.put((dist[target], target))
    # print(dist[target])
    # print(pq)
    while 0 != pq.qsize():
        curr_cost, curr = pq.get()
        # print(pq)
        if curr_cost > dist[curr]:
            continue
        visited.add(curr)
        #print(curr)
        for [v, curr] in G.in_edges(curr):
            if (G.edges[v, curr]["Balance"] + G.edges[curr, v]["Balance"] >= amount[curr]) and v not in visited:
                if(done[v] == 0 and G.nodes[v]["Tech"] == tech):
                    paths[v] = [v] + build_path(curr,prev)
                    done[v] = 1
                cost = dist[curr] + cost_function(G, amount[curr], curr, v)
                if cost < dist[v]:
                    dist[v] = cost
                    prev[v] = curr
                    delay[v] = G.edges[v, curr]["Delay"] + delay[curr]
                    amount[v] = amount[curr] + G.edges[v, curr]["BaseFee"] + amount[curr] * G.edges[v, curr]["FeeRate"]
                    # t = G.edges[v, curr]["LastFailure"]
                    # edge = edge_prob(t)
                    # prob[v] = edge * prob[curr]
                    pq.put((dist[v], v))
    return paths

def Eclair(G, source, target, amt, path=None):
    G1 = G.copy()
    B = nd.nested_dict()
    if (path == None):
        B[0],d,c,di = Dijkstra(G,source,target,amt,eclair_cost_fun)
        #print(B[0],di)
    # print(B[0]["Path"])
    else:
        B[0] = path
    paths = PriorityQueue()
    # leng = 0
    for k in range(1, 3):
        A = B[k - 1]
        for i in range(0, len(A) - 2):
            edges_removed = []
            spurnode = A[i]
            rootpath = A[0:i + 1]
            # print(spurnode,rootpath,len(B))
            for j in range(0, len(B)):
                p = B[j]
                # print(p,j)
                if rootpath == p[0:i + 1]:
                    if (G1.has_edge(p[i], p[i + 1])):
                        G1.remove_edge(p[i], p[i + 1])
                        G1.remove_edge(p[i + 1], p[i])
                        edges_removed.append((p[i], p[i + 1]))
                        edges_removed.append((p[i + 1], p[i]))
            for rootpathnode in rootpath:
                if (rootpathnode != spurnode):
                    if (G1.has_node(rootpathnode)):
                        for [u, v] in G1.copy().in_edges(rootpathnode):
                            G1.remove_edge(u, v)
                            G1.remove_edge(v, u)
                            edges_removed.append((u, v))
                            edges_removed.append((v, u))
            # print(eclair_single(G1,spurnode,target,amt))
            spurpath, delay, cost, dist = Dijkstra(G1, spurnode, target, amt,eclair_cost_fun)
            totalpath = rootpath + spurpath[1:]

            flag = 0
            if totalpath == rootpath:
                flag = 1
            # for t in range(0, len(paths)):
            #     if (totalpath == paths[t]["Path"]):
            #         flag = 1
            if flag == 0:
                # paths[leng]["Path"] = totalpath
                di = calc_params(G, totalpath, amt)
                # paths[leng]["visited"] = 0
                # leng += 1
                paths.put((di, totalpath))
            for e in edges_removed:
                u, v = e
                G1.add_edge(u, v)
                G1.edges[u, v]["Delay"] = G.edges[u, v]["Delay"]
                G1.edges[u, v]["Balance"] = G.edges[u, v]["Balance"]
                G1.edges[u, v]["BaseFee"] = G.edges[u, v]["BaseFee"]
                G1.edges[u, v]["FeeRate"] = G.edges[u, v]["FeeRate"]
                G1.edges[u, v]["Age"] = G.edges[u, v]["Age"]
        flag1 = 0
        while flag1 == 0:
            d, p = paths.get()
            flag2 = 0
            for i in range(0, len(B)):
                if p == B[i]:
                    flag2 = 1
                    break
            if flag2 == 0:
                flag1 = 1
        if p == None:
            break
        else:
            # minpath = paths[0]["Path"]
            # mincost = paths[0]["Dist"]
            # index = -1
            # for i in range(1, leng):
            #     if mincost > paths[i]["Dist"] and paths[i]["visited"] == 0:
            #         mincost = paths[i]["Dist"]
            #         minpath = paths[i]["Path"]
            #         index = i
            # paths[index]["visited"] = 1
            B[k] = p
            #print(minpath,mincost)
    # print(B[0]["Path"], B[1]["Path"], B[2]["Path"])
    #print(len(B))
    return B


def modifiedEclair(G, source, target, amt, path=None):
    G1 = G.copy()
    B = nd.nested_dict()
    if (path == None):
        B[0],d,c,di = Dijkstra(G,source,target,amt,eclair_cost_fun)
        #print(B[0],di)
    # print(B[0]["Path"])
    else:
        B[0] = path
    paths = PriorityQueue()
    leng = 0
    for k in range(1, 3):
        A = B[k - 1]
        for i in range(len(A)-1, 0,-1):
            edges_removed = []
            spurnode = A[i]
            rootpath = A[i:len(A)]
            # print(spurnode,rootpath,len(B))
            for j in range(0, len(B)):
                p = B[j]
                # print(p,j)
                if rootpath == p[i:len(A)]:
                    if (G1.has_edge(p[i], p[i-1])):
                        G1.remove_edge(p[i], p[i-1])
                        G1.remove_edge(p[i-1], p[i])
                        edges_removed.append((p[i], p[i-1]))
                        edges_removed.append((p[i-1], p[i]))
            for rootpathnode in rootpath:
                if (rootpathnode != spurnode):
                    if (G1.has_node(rootpathnode)):
                        for [u, v] in G1.copy().in_edges(rootpathnode):
                            G1.remove_edge(u, v)
                            G1.remove_edge(v, u)
                            edges_removed.append((u, v))
                            edges_removed.append((v, u))
            # print(eclair_single(G1,spurnode,target,amt))
            spurpath, delay, cost, dist = Dijkstra(G1, source,spurnode, amt,eclair_cost_fun)
            totalpath = spurpath[:-1]+rootpath

            flag = 0
            if totalpath == rootpath:
                flag = 1
            # for t in range(0, len(paths)):
            #     if (totalpath == paths[t]["Path"]):
            #         flag = 1
            if flag == 0:
                # paths[leng]["Path"] = totalpath
                di = calc_params(G, totalpath, amt)
                # paths[leng]["visited"] = 0
                # leng += 1
                paths.put((di,totalpath))
            for e in edges_removed:
                u, v = e
                G1.add_edge(u, v)
                G1.edges[u, v]["Delay"] = G.edges[u, v]["Delay"]
                G1.edges[u, v]["Balance"] = G.edges[u, v]["Balance"]
                G1.edges[u, v]["BaseFee"] = G.edges[u, v]["BaseFee"]
                G1.edges[u, v]["FeeRate"] = G.edges[u, v]["FeeRate"]
                G1.edges[u, v]["Age"] = G.edges[u, v]["Age"]
        flag1 = 0
        while flag1 == 0:
            d, p = paths.get()
            flag2 = 0
            for i in range(0, len(B)):
                if p == B[i]:
                    flag2 = 1
                    break
            if flag2 == 0:
                flag1 = 1
        if p == None:
            break
        else:
            # minpath = paths[0]["Path"]
            # mincost = paths[0]["Dist"]
            # index = -1
            # for i in range(1, leng):
            #     if mincost > paths[i]["Dist"] and paths[i]["visited"] == 0:
            #         mincost = paths[i]["Dist"]
            #         minpath = paths[i]["Path"]
            #         index = i
            # paths[index]["visited"] = 1
            B[k] = p
            #print(minpath,mincost)
    # print(B[0]["Path"], B[1]["Path"], B[2]["Path"])
    #print(len(B))
    return B

def Dijkstra_general(G,source,target,amt,cost_function):
    paths = {}
    paths1 = {}
    paths2 = {}
    dist = {}
    dist1 = {}
    dist2  = {}
    delay = {}
    delay1 = {}
    delay2 = {}
    amount = {}
    amount1 = {}
    amount2 = {}
    visited = {}
    for node in G.nodes():
        amount[node] = -1
        amount1[node] = -1
        amount2[node] = -1
        delay[node] = -1
        delay1[node] = -1
        delay2[node] = -1
        dist[node] = inf
        dist1[node] = inf
        dist2[node] = inf
        visited[node] = 0
        paths[node] = []
        paths1[node] = []
        paths2[node] = []
    prev = {}
    pq = PriorityQueue()
    dist[target] = 0
    dist1[target] = 0
    dist2[target] = 0
    delay[target] = 0
    delay1[target] = 0
    delay2[target] = 0
    # print(dist[target])
    paths[target] = [target]
    paths1[target] = [target]
    paths2[target] = [target]
    # print(dist[target])
    amount[target] = amt
    amount1[target] = amt
    amount2[target] = amt
    # print(dist[target],amount[target])
    pq.put((dist[target], target))
    # print(dist[target])
    # print(pq)
    k = 0
    path = {}
    while 0 != pq.qsize():
        curr_cost, curr = pq.get()
        # print(curr)
        # print(pq)
        if curr_cost > dist2[curr]:
            continue
        if visited[curr] == 0:
            p = paths[curr]
            d = delay[curr]
            a = amount[curr]
            di = dist[curr]
        elif visited[curr] == 1:
            p = paths1[curr]
            d = delay1[curr]
            a = amount1[curr]
            di = dist1[curr]
        elif visited[curr] == 2:
            p = paths2[curr]
            d = delay2[curr]
            a = amount2[curr]
            di = dist2[curr]
        visited[curr]+=1
        for [v, curr] in G.in_edges(curr):
            if v == source and G.edges[v, curr]["Balance"] >= a:
                #return [v] + paths[curr], delay[curr] + G.edges[v, curr]["Delay"], amount[curr], dist[curr]
                path[k]= [v] + p
                k+=1
                if k == 3:
                    # print(path)
                    return path
            if (G.edges[v, curr]["Balance"] + G.edges[curr, v]["Balance"] >= a) and visited[v]<3 and v not in p:
                cost = di + cost_function(G, a, curr, v)

                if cost < dist[v]:
                    dist2[v] = dist1[v]
                    paths2[v] = paths1[v]
                    delay2[v] = delay1[v]
                    amount2[v] = amount1[v]
                    dist1[v] = dist[v]
                    paths1[v] = paths[v]
                    delay1[v] = delay[v]
                    amount1[v] = amount[v]
                    dist[v] = cost
                    paths[v] = [v] + p
                    delay[v] = G.edges[v, curr]["Delay"] + d
                    amount[v] = a + G.edges[v, curr]["BaseFee"] + a * G.edges[v, curr]["FeeRate"]
                    pq.put((dist[v], v))
                elif cost < dist1[v]:
                    dist2[v] = dist1[v]
                    paths2[v] = paths1[v]
                    delay2[v] = delay1[v]
                    amount2[v] = amount1[v]
                    dist1[v] = cost
                    paths1[v] = [v] + p
                    delay1[v] = G.edges[v, curr]["Delay"] + d
                    amount1[v] = a + G.edges[v, curr]["BaseFee"] + a * G.edges[v, curr]["FeeRate"]
                    pq.put((dist1[v], v))
                elif cost < dist2[v]:
                    dist2[v] = cost
                    paths2[v] = [v] + p
                    delay2[v] = G.edges[v, curr]["Delay"] + d
                    amount2[v] = a + G.edges[v, curr]["BaseFee"] + a * G.edges[v, curr]["FeeRate"]
                    pq.put((dist2[v], v))
    return [], -1, -1, -1

def Dijkstra_general_all_paths(G,target,amt,cost_function):
    paths = {}
    paths1 = {}
    paths2 = {}
    path = {}
    path1 = {}
    path2 = {}
    dist = {}
    dist1 = {}
    dist2  = {}
    delay = {}
    delay1 = {}
    delay2 = {}
    amount = {}
    amount1 = {}
    amount2 = {}
    visited = {}
    done = {}
    for node in G.nodes():
        amount[node] = -1
        amount1[node] = -1
        amount2[node] = -1
        delay[node] = -1
        delay1[node] = -1
        delay2[node] = -1
        dist[node] = inf
        dist1[node] = inf
        dist2[node] = inf
        visited[node] = 0
        paths[node] = []
        paths1[node] = []
        paths2[node] = []
        done[node] = 0
    prev = {}
    pq = PriorityQueue()
    dist[target] = 0
    dist1[target] = 0
    dist2[target] = 0
    delay[target] = 0
    delay1[target] = 0
    delay2[target] = 0
    # print(dist[target])
    paths[target] = [target]
    paths1[target] = [target]
    paths2[target] = [target]
    # print(dist[target])
    amount[target] = amt
    amount1[target] = amt
    amount2[target] = amt
    # print(dist[target],amount[target])
    pq.put((dist[target], target))
    # print(dist[target])
    # print(pq)
    while 0 != pq.qsize():
        curr_cost, curr = pq.get()
        # print(curr)
        # print(pq)
        if curr_cost > dist2[curr]:
            continue
        if visited[curr] == 0:
            p = paths[curr]
            d = delay[curr]
            a = amount[curr]
            di = dist[curr]
        elif visited[curr] == 1:
            p = paths1[curr]
            d = delay1[curr]
            a = amount1[curr]
            di = dist1[curr]
        elif visited[curr] == 2:
            p = paths2[curr]
            d = delay2[curr]
            a = amount2[curr]
            di = dist2[curr]
        visited[curr]+=1
        for [v, curr] in G.in_edges(curr):
            if done[v] == 0 and G.nodes[v]["Tech"] == 2:
                path[v] = [v] + p
                done[v] = 1
            elif done[v] == 1 and G.nodes[v]["Tech"] == 2:
                path1[v] = [v] + p
                done[v] = 2
            elif done[v] == 2 and G.nodes[v]["Tech"] == 2:
                path2[v] = [v] + p
                done[v] = 3
            if (G.edges[v, curr]["Balance"] + G.edges[curr, v]["Balance"] >= a) and visited[v]<3 and v not in p:
                cost = di + cost_function(G, a, curr, v)

                if cost < dist[v]:
                    dist2[v] = dist1[v]
                    paths2[v] = paths1[v]
                    delay2[v] = delay1[v]
                    amount2[v] = amount1[v]
                    dist1[v] = dist[v]
                    paths1[v] = paths[v]
                    delay1[v] = delay[v]
                    amount1[v] = amount[v]
                    dist[v] = cost
                    paths[v] = [v] + p
                    delay[v] = G.edges[v, curr]["Delay"] + d
                    amount[v] = a + G.edges[v, curr]["BaseFee"] + a * G.edges[v, curr]["FeeRate"]
                    pq.put((dist[v], v))
                elif cost < dist1[v]:
                    dist2[v] = dist1[v]
                    paths2[v] = paths1[v]
                    delay2[v] = delay1[v]
                    amount2[v] = amount1[v]
                    dist1[v] = cost
                    paths1[v] = [v] + p
                    delay1[v] = G.edges[v, curr]["Delay"] + d
                    amount1[v] = a + G.edges[v, curr]["BaseFee"] + a * G.edges[v, curr]["FeeRate"]
                    pq.put((dist1[v], v))
                elif cost < dist2[v]:
                    dist2[v] = cost
                    paths2[v] = [v] + p
                    delay2[v] = G.edges[v, curr]["Delay"] + d
                    amount2[v] = a + G.edges[v, curr]["BaseFee"] + a * G.edges[v, curr]["FeeRate"]
                    pq.put((dist2[v], v))
    return path,path1,path2