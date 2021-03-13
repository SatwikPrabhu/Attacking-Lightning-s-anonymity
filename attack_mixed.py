import networkx as nx
import pathFind as pf
import nested_dict as nd
import random as rn
import csv
from queue import PriorityQueue
from math import inf

max = inf
CBR = 648601
LND_RISK_FACTOR = 0.000000015
C_RISK_FACTOR = 10
RISK_BIAS = 1
DEFAULT_FUZZ = 0.05
MIN_DELAY = 9
MAX_DELAY = 2016
MIN_CAP = 1
MAX_CAP = 100000000
MIN_AGE = 505149
MAX_AGE = CBR
DELAY_RATIO = 0.15
CAPACITY_RATIO = 0.5
AGE_RATIO = 0.35

# Function to normalize a a value between max and min - used by Eclair
def normalize(val, min, max):
    if val <= min:
        return 0.00001
    if val > max:
        return 0.99999
    return (val - min) / (max - min)

# Function that identifies each potential destination based on a depth first search and the potential senders for each destination.
# The search begins from the next node and considers all nodes that have loopless paths the given delay from the next node(excluding the previous node and the adversary)
# as potential destinations.
# For each potential destination, the set of sources that would use the same sub-path that we found during the search using either lnd, c-lightning or eclair
# by calling the respective deanoniyize functions.
def dest_reveal_new(G,adversary,delay,amount,pre,next):
    T = nd.nested_dict()
    flag1 = True
    anon_sets = nd.nested_dict()
    level = 0
    index = 0
    # Level 0 only contains the next node
    T[0]["nodes"] = [next]
    T[0]["delays"] = [delay]
    print(delay)
    T[0]["previous"] = [-1]
    T[0]["visited"] = [[pre,adversary,next]]
    T[0]["amounts"] = [amount]
    x = -1

    paths = nd.nested_dict()
    num_paths = 0
    # flag to indicate that going further would result only in invalid nodes as the delay limit is reached for all nodes in the current level
    flag = True

    while(flag):
        level+=1
        # Stop when level is greater than 3 - it takes forever otherwise
        if(level == 4):
            flag1 = False
            break
        t1 = T[level - 1]["nodes"]
        d1 = T[level - 1]["delays"]
        p1 = T[level - 1]["previous"]
        v1 = T[level - 1]["visited"]
        a1 = T[level - 1]["amounts"]
        pr1 = T[level - 1]["probs"]
        t2 = []
        d2 = []
        p2 = []
        v2 = [[]]
        a2 = []
        pr2 = []
        for i in range(0,len(t1)):
            u = t1[i]
            for [u,v] in G.out_edges(u):
                # Checks if v is not repeating in the same path, delay limit is not reached after visiting v and the capacity condition is true after deducting fees
                if(v!=pre and v!=adversary  and v!=next and v not in v1[i] and (d1[i] - G.edges[u,v]["Delay"])>=0 and (G.edges[u,v]["Balance"]+G.edges[v,u]["Balance"])>=((a1[i] - G.edges[u, v]["BaseFee"]) / (1 + G.edges[u, v]["FeeRate"]))):
                    t2.append(v)
                    d2.append(d1[i] - G.edges[u,v]["Delay"])
                    p2.append(i)
                    v2.append(v1[i]+[v])
                    a2.append(((a1[i] - G.edges[u, v]["BaseFee"]) / (1 + G.edges[u, v]["FeeRate"])))
          
        T[level]["nodes"] = t2
        #print(level,t2,d2)
        T[level]["delays"] = d2
        T[level]["previous"] = p2
        T[level]["visited"] = v2
        T[level]["amounts"] = a2
        #T[level]["probs"] = pr2
        #print(t2,d2,p2)
        print(level,len(t2))
        # Stop if the current level has 0 nodes
        if(len(t2) == 0):
            flag = False
    level = level - 1
    while(level>=0):
        t = T[level]["nodes"]
        d = T[level]["delays"]
        p = T[level]["previous"]
        a = T[level]["amounts"]
        v = T[level]["visited"]
        #print(level)
        for i in range(0, len(t)):
            # Potential destination if delay is 0
            if(d[i] == 0):
                #construct the path found from the next node to the destination
                path = []
                level1 = level
                path.append(T[level1]["nodes"][i])
                loc = T[level1]["previous"][i]
                while (level1 > 0):
                    level1 = level1 - 1
                    path.append(T[level1]["nodes"][loc])
                    loc = T[level1]["previous"][loc]
                path.reverse()
                # Add pre and adversary to the start of the path
                path = [pre,adversary]+path
                # Double check that path is loop free
                if (len(path) == len(set(path))):
                    #print(path, level)
                    amt = a[i]
                    pot = path[len(path) - 1]
                    # For each destination find the sources that would use this subpath using either lnd,c-lightning or eclair
                    sources_lnd = deanonymize_lnd(G,pot,path,amt)
                    if sources_lnd != []:
                        print("match",pot,"lnd")
                        anon_sets[pot]["lnd"] = list(sources_lnd)
                    # Check for more fuzz values only if the anonymity sets do not match for fuzz values -1 and 1
                    fuzz = -0.8
                    sources_c = deanonymize_c(G,pot,path,amt,-1)
                    sources_c1 = deanonymize_c(G,pot,path,amt,1)
                    if(sources_c1!=sources_c):
                        sources_c = sources_c + sources_c1
                        while fuzz<=0.8:
                            s = deanonymize_c(G,pot,path,amt,fuzz)
                            if(s!=[]):
                                sources_c = sources_c + s
                            fuzz+=0.2
                    sources_c = list(set(sources_c))
                    if sources_c != []:
                        print("match",pot,"c",fuzz)
                        anon_sets[pot]["c"] = list(set(sources_c))
                    sources_ecl = deanonymize_ecl(G,pot,path,amt)

                    if sources_ecl != []:
                        print("match",pot,"ecl")
                        anon_sets[pot]["ecl"] = list(sources_ecl)
        level = level - 1
    return anon_sets,flag1

# Returns potential sources that would use lnd to reach target using the subpath found. The key idea here is that we implement all sources Dijkstra for the given target
# and the transaction amount. If at any point we find a discrepency between the found path and the path constructed using dijkstra for any node in the sub-path found, 
# the target cannot be the real destination. If the complete sub-path matches the dijkstra condition, we proceed to find all possible sources that use this samme sub-path.
def deanonymize_lnd(G,target,path,amt):
    pq = PriorityQueue()
    delays = {}
    costs = {}
    paths = nd.nested_dict()
    paths1 = nd.nested_dict()
    dists = {}
    dists1 = {}
    delays1 = {}
    costs1 = {}
    visited = set()
    previous = {}
    done = {}
    # prob = {}
    sources = []
    pre = path[0]
    adv = path[1]
    nxt = path[2]
    for node in G.nodes():
        previous[node] = -1
        delays[node] = -1
        costs[node] = max
        paths[node] = []
        dists[node] = max
        done[node] = 0
        paths1[node] = []
        # prob[node] = 1
        dists1[node] = max
        delays1[node] = -1
        costs1[node] = max
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
                if G.nodes[v]["Tech"] == 0:
                    cost = dists[curr] + costs[curr] * G.edges[v, curr]["Delay"] * LND_RISK_FACTOR
                    if cost < dists1[v]:
                        paths1[v] = [v]+paths[curr]
                        dists1[v] = cost
                        delays1[v] = delays[curr] + G.edges[v,curr]["Delay"]
                        costs1[v] = costs[curr]
                cost = dists[curr]+ pf.lnd_cost_fun(G,costs[curr],curr,v)
                if cost < dists[v]:
                    paths[v] = [v]+paths[curr]
                    dists[v] = cost
                    delays[v] = delays[curr] + G.edges[v,curr]["Delay"]
                    costs[v] = costs[curr] + G.edges[v, curr]["BaseFee"] + costs[curr] * G.edges[v, curr]["FeeRate"]
                    pq.put((dists[v],v))
        # If at any point the sub-path found is not found to be optimal, this is definetely not the destination if using lnd since the sub-path from an intermediary to 
        # the destination has to be the cheapest path from the intermediary to the destination.
        if(curr in path[1:]):
            ind = path.index(curr)
            if(paths[curr]!=path[ind:]):
                return []
            if curr == adv:
                flag1 = 1
        if(curr == pre):
            # If pre is the source, the path from pre need to not match the path found since, the cost from the source to the second node is computed differently.
            # Moreover, the source would not choose the absolute cheapest path since the first hop may not have sufficient forward balance. 
            # Thus, pre has to be the source if the paths dont match, since the paths would only match if pre is an intermediary.
            if paths[pre] != path:
                return [pre]
            else:
                # if the paths do match, pre is just one possible source
                sources.append(pre)
            flag2 = 1
        if flag1 == 1 and flag2 == 1:
            # since if pre is in the path from curr, the path from pre has to match the path we had found as it is the cheapest path from pre. This measns that curr
            # is a valid second node. So, all neighbors of curr that have not occured in the path are potential sources.
            if pre in paths[curr]:
                for [v,curr] in G.in_edges(curr):
                        if v not in paths[curr]:
                            sources.append(v)
    sources = list(set(sources))
    return sources

# Returns potential sources that would use lnd to reach target using the subpath found. The key idea here is that we implement all sources Dijkstra for the given target
# and the transaction amount. If at any point we find a discrepency between the found path and the path constructed using dijkstra for any node in the sub-path found, 
# the target cannot be the real destination. If the complete sub-path matches the dijkstra condition, we proceed to find all possible sources that use this samme sub-path.
def deanonymize_c(G,target,path,amt,fuzz):
    pq = PriorityQueue()
    cost_function = pf.c_cost_fun(fuzz)
    delays = {}
    costs = {}
    paths = nd.nested_dict()
    paths1 = nd.nested_dict()
    dists = {}
    visited = set()
    previous = {}
    done = {}
    # prob = {}
    sources = []
    pre = path[0]
    adv = path[1]
    nxt = path[2]
    for node in G.nodes():
        previous[node] = -1
        delays[node] = -1
        costs[node] = max
        paths[node] = []
        dists[node] = max
        done[node] = 0
        paths1[node] = []
        # prob[node] = 1
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
                if done[v] == 0 and G.nodes[v]["Tech"] == 1:
                    paths1[v] = [v]+paths[curr]
                    done[v] = 1
                cost = dists[curr]+ cost_function(G,costs[curr],curr,v)
                if cost < dists[v]:
                    paths[v] = [v]+paths[curr]
                    dists[v] = cost
                    delays[v] = delays[curr] + G.edges[v,curr]["Delay"]
                    costs[v] = costs[curr] + G.edges[v, curr]["BaseFee"] + costs[curr] * G.edges[v, curr]["FeeRate"]
                    # prob[v] = pf.edge_prob(G.edges[v,curr]["LastFailure"])*prob[curr]
                    pq.put((dists[v],v))
        # If at any point the sub-path found is not found to be optimal, this is definetely not the destination if using lnd since the sub-path from an intermediary to 
        # the destination has to be the cheapest path from the intermediary to the destination.
        if(curr in path[1:]):
            ind = path.index(curr)
            if(paths[curr]!=path[ind:]):
                return []
            if curr == adv:
                flag1 = 1
        if(curr == pre):
            # If pre is the source, the path from pre need to not match the path found since, the cost from the source to the second node is computed differently.
            # Moreover, the source would not choose the absolute cheapest path since the first hop may not have sufficient forward balance. 
            # Thus, pre has to be the source if the paths dont match, since the paths would only match if pre is an intermediary.
            if paths[pre] != path:
                return [pre]
            else:
                # if the paths do match, pre is just one possible source
                sources.append(pre)
            flag2 = 1
        if flag1 == 1 and flag2 == 1:
            # since if pre is in the path from curr, the path from pre has to match the path we had found as it is the cheapest path from pre. This measns that curr
            # is a valid second node. So, all neighbors of curr that have not occured in the path are potential sources.
            if pre in paths[curr]:
                for [v,curr] in G.in_edges(curr):
                        if v not in paths[curr]:
                            sources.append(v)
    sources = list(set(sources))
    return sources

# Returns potential sources that would use lnd to reach target using the subpath found. 
def deanonymize_ecl(G,target,pa,amt):
    paths = {}
    paths1 = {}
    paths2 = {}
    path = {}
    path1 = {}
    path2 = {}
    dist = {}
    dist1 = {}
    dist2 = {}
    delay = {}
    delay1 = {}
    delay2 = {}
    amount = {}
    amount1 = {}
    amount2 = {}
    visited = {}
    done = {}
    pre = pa[0]
    adv = pa[1]
    nxt = pa[2]
    sources = []
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
    flag1 = 0
    flag2 = 0
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
        visited[curr] += 1
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
            if (G.edges[v, curr]["Balance"] + G.edges[curr, v]["Balance"] >= a) and visited[v] < 3 and v not in p:
                cost = di + pf.eclair_cost_fun(G, a, curr, v)

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
        # If none of the three best paths satisfy the sub-path found then the destination cannot be reached using Eclair
        if(curr in pa[1:]):
            ind = pa.index(curr)
            if visited[curr] == 3:
                if (paths[curr] != pa[ind:] and paths1[curr] != pa[ind:] and paths2[curr] != pa[ind:]):
                    return []
            # If we find a match, then we do not need to look for other paths from curr
            if pa[ind:] == p:
                visited[curr] = 3
                if curr == adv:
                    # print(paths[curr],paths1[curr],paths2[curr])
                    flag1 = 1
        if(curr == pre):
            if pa == p or visited[curr] == 3:
                visited[curr] = 3
                 # If pre is not an intermediary, then it must be the source
            if paths[pre] != pa and paths1[pre]!=pa and paths2[pre]!=pa:
                if G.nodes[pre]["Tech"] != 2:
                    return []
                return [pre]
            else:
                # pre could still be a source
                if G.nodes[pre]["Tech"] == 2:
                    sources.append(pre)
                flag2 = 1
            # print(paths[curr], paths1[curr], paths2[curr])
        if flag1 == 1 and flag2 == 1:
                # fill remaining possible sources
                if pre in p:
                    for [v, curr] in G.in_edges(curr):
                        if v not in paths[curr] and G.nodes[v]["Tech"] == 2:
                            sources.append(v)
#                     ind = p.index(pre)
#                     if p[ind:] == pa:
#                         visited[curr] = 3
#                         for [v, curr] in G.in_edges(curr):
#                             if v not in paths[curr] and G.nodes[v]["Tech"] == 2:
#                                 sources.append(v)
#                     else:
#                         print("error")
    sources = list(set(sources))
    return sources
