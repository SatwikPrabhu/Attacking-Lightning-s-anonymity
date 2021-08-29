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


def normalize(val, min, max):
    if val <= min:
        return 0.00001
    if val > max:
        return 0.99999
    return (val - min) / (max - min)
def tr_amt(G,path,amt):
    for i in range(0,len(path)-1):
        amt = ((amt - G.edges[path[i], path[i + 1]]["BaseFee"]) / (1 + G.edges[path[i], path[i + 1]]["FeeRate"]))
    return amt





def dest_reveal_new(G,adversary,delay,amount,pre,next):
    T = nd.nested_dict()
    flag1 = True
    anon_sets = nd.nested_dict()
    level = 0
    index = 0
    T[0]["nodes"] = [next]
    T[0]["delays"] = [delay]
    #print(delay)
    T[0]["previous"] = [-1]
    T[0]["visited"] = [[pre,adversary,next]]
    T[0]["amounts"] = [amount]
    #pr = pf.edge_prob(G.edges[pre,adversary]["LastFailure"])*pf.edge_prob(G.edges[adversary,next]["LastFailure"])
    #T[0]["probs"] = [pr]
    x = -1
    # if T[0]["delays"][0] == 0:
        # maybe_targets[index]["target"] = next
        # maybe_targets[index]["path"] = [adversary,next]
        # maybe_targets[index]["delay"] = delay
        # maybe_targets[index]["amt"] = amount
        # maybe_targets[index]["tech"] = 0
        # maybe_targets[index]["sources"] = source_reveal(G, [pre, adversary,next], 0, 0, amount, pre, next,adversary)
        # index += 1
        # paths = pf.Dijkstra_all_paths(G,next,amount,pf.lnd_cost_fun)
        # for u in paths:
        #     if pre in paths[u]:
        #         ind = paths[u].index(pre)
        #         if(paths[u][ind:] == [pre,adversary,next]):
        #             anon_sets[index] = [u,next]
        #             print("match",u,next)

    paths = nd.nested_dict()
    num_paths = 0
    flag = True

    while(flag):
        level+=1
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
            # if v1[i] == []:
            #     print(u)
            #     print("yes",u)
            for [u,v] in G.out_edges(u):
                #print(v)
                # p = p1[i]
                # flag1 = 0
                # level2 = level - 2
                # while(level2>=1):
                #     if(T[level2]["nodes"][p] == v):
                #         flag1 = 1
                #         break
                #     else:
                #         p = T[level2]["previous"][p]
                #         level2 = level2 - 1
                #pr = pf.edge_prob(G.edges[u,v]["LastFailure"])*pr1[i]
                if(v!=pre and v!=adversary  and v!=next and v not in v1[i] and (d1[i] - G.edges[u,v]["Delay"])>=0 and (G.edges[u,v]["Balance"]+G.edges[v,u]["Balance"])>=((a1[i] - G.edges[u, v]["BaseFee"]) / (1 + G.edges[u, v]["FeeRate"]))):
                    t2.append(v)
                    d2.append(d1[i] - G.edges[u,v]["Delay"])
                    p2.append(i)
                    v2.append(v1[i]+[v])
                    a2.append(((a1[i] - G.edges[u, v]["BaseFee"]) / (1 + G.edges[u, v]["FeeRate"])))
                    #pr2.append(pr)
        T[level]["nodes"] = t2
        #print(level,t2,d2)
        T[level]["delays"] = d2
        T[level]["previous"] = p2
        T[level]["visited"] = v2
        T[level]["amounts"] = a2
        #T[level]["probs"] = pr2
        #print(t2,d2,p2)
        #print(level,len(t2))
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
        # if(level == 0):
        #     print(t,d)
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
                    #print(path, level)
                    amt = a[i]
                    pot = path[len(path) - 1]
                    # if pot == 10515:
                    #     print(path,level)
                    sources_lnd = deanonymize_lnd(G,pot,path,amt)
                    if sources_lnd != []:
                        #print("match",pot,"lnd")
                        anon_sets[pot][0] = list(sources_lnd)
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
                     # sources_c = deanonymize_c(G,pot,path,amt,-1)
                    if sources_c != []:
                        #print("match",pot,"c",fuzz)
                        anon_sets[pot][1] = list(set(sources_c))
                    # if pot == 10892:
                    #     print(pot,"yes",amt)
                    sources_ecl = deanonymize_ecl(G,pot,path,amt)

                    if sources_ecl != []:
                        #print("match",pot,"ecl")
                        anon_sets[pot][2] = list(sources_ecl)
                    # if paths == [pre, adversary] + path:
                    #     maybe_targets[index]["target"] = pot
                    #     maybe_targets[index]["path"] = [adversary] + path
                    #     maybe_targets[index]["delay"] = delay
                    #     maybe_targets[index]["amt"] = amt
                    #     maybe_targets[index]["tech"] = 0
                    #     maybe_targets[index]["sources"] = source_reveal(G, [pre, adversary] + path, 0, 0, amt, pre, next,
                    #                                                     adversary)
                    #     index += 1
                    # for u in paths:
                    #     if pre in paths[u]:
                    #         ind = paths[u].index(pre)
                    #         if paths[u][ind:] == [pre,adversary] + path:
                    #             anon_sets[index] = [u,pot]
                    #             index+=1
                    #             print("match",u,pot)
        level = level - 1
    return anon_sets,flag1

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
                    # prob[v] = pf.edge_prob(G.edges[v,curr]["LastFailure"])*prob[curr]
                    pq.put((dists[v],v))
        if(curr in path[1:]):
            ind = path.index(curr)
            if(paths[curr]!=path[ind:]):
                return []
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
    sources = list(set(sources))
    return sources

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
        if(curr in path[1:]):
            ind = path.index(curr)
            if(paths[curr]!=path[ind:]):
                return []
            if curr == adv:
                flag1 = 1
        # if flag1 == 1:
        #     print("path", paths[adv])
        if (curr == pre):
            if paths[pre] != path:
                return [pre]
            else:
                sources.append(pre)
            flag2 = 1
        if flag1 == 1 and flag2 == 1:
            if pre in paths[curr]:
                for [v, curr] in G.in_edges(curr):
                    if v not in paths[curr]:
                        sources.append(v)
    sources = list(set(sources))
    return sources

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
        # if curr == 9591 or curr == 2417:
        #     if target == 10515:
        #         print(curr,p)
        # if(curr == pre or curr ==adv):
        #     print(p,curr)
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
            # if v == 6963 and target == 10515:
            #     print([v]+p)
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
        # if (curr == 2336 or curr == 7728 or curr == 4146):
        #     print(curr, p)
        if(curr in pa[1:]):
            ind = pa.index(curr)
            if visited[curr] == 3:
                if (paths[curr] != pa[ind:] and paths1[curr] != pa[ind:] and paths2[curr] != pa[ind:]):
                    return []
            if pa[ind:] == p:
                visited[curr] = 3
                if curr == adv:
                    # print(paths[curr],paths1[curr],paths2[curr])
                    flag1 = 1
        if(curr == pre):
            if pa == p or visited[curr] == 3:
                visited[curr] = 3
                if paths[pre] != pa and paths1[pre] != pa and paths2[pre] != pa:
                    return [pre]
                else:
                    # pre could still be a source
                    sources.append(pre)
                    flag2 = 1
                    # print(paths[curr], paths1[curr], paths2[curr])
        if flag1 == 1 and flag2 == 1:
            # fill remaining possible sources
            if pre in p:
                for [v, curr] in G.in_edges(curr):
                    if v not in p:
                        sources.append(v)
                # if visited[curr] == 3:
                #     if G.nodes[pre]["Tech"] == 2:
                #         sources.append(pre)
                #     er = 0
                #     if pre in paths[curr]:
                #         ind = paths[curr].index(pre)
                #         if paths[curr][ind:] == pa:
                #         #     print("error")
                #         # else:
                #             for [v,curr] in G.in_edges(curr):
                #                 if v not in paths[curr] and G.nodes[v]["Tech"] == 2:
                #                     sources.append(v)
                #         else:
                #             er+=1
                #     if pre in paths1[curr]:
                #         ind = paths1[curr].index(pre)
                #         if paths1[curr][ind:] == pa:
                #         #     print("error")
                #         # else:
                #             for [v,curr] in G.in_edges(curr):
                #                 if v not in paths1[curr] and G.nodes[v]["Tech"] == 2:
                #                     sources.append(v)
                #         else:
                #             er+=1
                #     if pre in paths2[curr]:
                #         ind = paths2[curr].index(pre)
                #         if paths2[curr][ind:] == pa:
                #         #     print("error")
                #         # else:
                #             for [v,curr] in G.in_edges(curr):
                #                 if v not in paths2[curr] and G.nodes[v]["Tech"] == 2:
                #                     sources.append(v)
                #         else:
                #             er+=1
                #     if er == 3:
                #         print("error",curr)
    sources = list(set(sources))
    return sources