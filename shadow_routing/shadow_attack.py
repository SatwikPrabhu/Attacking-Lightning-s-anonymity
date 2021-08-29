import networkx as nx
import pathFind as pf
import nested_dict as nd
import random as rn
import csv
from queue import PriorityQueue
from math import inf

max = inf
LND_RISK_FACTOR = 0.000000015

def tr_amt(G,path,amt):
    for i in range(0,len(path)-1):
        amt = ((amt - G.edges[path[i], path[i + 1]]["BaseFee"]) / (1 + G.edges[path[i], path[i + 1]]["FeeRate"]))
    return amt

def source_reveal(G,path,amt,pre):
    maybe_source = [pre]
    paths = pf.Dijkstra_all_paths(G,path[len(path)-1],amt,pf.lnd_cost_fun)
    for u in paths:
        if(pre in paths[u]):
            path1 = paths[u]
            index = path1.index(pre)
            if (path1[index:] == path):
                maybe_source.append(u)
                #maybe_source.append(u)
    return maybe_source

def dest_reveal_new(G,adversary,delay,amount,pre,next):
    T = nd.nested_dict()
    flag1 = True
    anon_sets = nd.nested_dict()
    level = 0
    index = 0
    T[0]["nodes"] = [next]
    T[0]["delays"] = [delay]
    print(delay)
    T[0]["previous"] = [-1]
    T[0]["visited"] = [[pre,adversary,next]]
    T[0]["amounts"] = [amount]
    # pr = pf.edge_prob(G.edges[pre,adversary]["LastFailure"])*pf.edge_prob(G.edges[adversary,next]["LastFailure"])
    # T[0]["probs"] = [pr]
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
        if(level == 3):
            flag1 = False
            break
        t1 = T[level - 1]["nodes"]
        d1 = T[level - 1]["delays"]
        p1 = T[level - 1]["previous"]
        v1 = T[level - 1]["visited"]
        a1 = T[level - 1]["amounts"]
        # pr1 = T[level - 1]["probs"]
        t2 = []
        d2 = []
        p2 = []
        v2 = [[]]
        a2 = []
        # pr2 = []
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
                # pr = pf.edge_prob(G.edges[u,v]["LastFailure"])*pr1[i]
                if(v!=pre and v!=adversary  and v!=next and v not in v1[i]  and (G.edges[u,v]["Balance"]+G.edges[v,u]["Balance"])>=((a1[i] - G.edges[u, v]["BaseFee"]) / (1 + G.edges[u, v]["FeeRate"]))):
                    t2.append(v)
                    d2.append(d1[i] - G.edges[u,v]["Delay"])
                    p2.append(i)
                    v2.append(v1[i]+[v])
                    a2.append(((a1[i] - G.edges[u, v]["BaseFee"]) / (1 + G.edges[u, v]["FeeRate"])))
                    # pr2.append(pr)
        T[level]["nodes"] = t2
        #print(level,t2,d2)
        T[level]["delays"] = d2
        T[level]["previous"] = p2
        T[level]["visited"] = v2
        T[level]["amounts"] = a2
        # T[level]["probs"] = pr2
        #print(t2,d2,p2)
        print(level,len(t2))
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
            #if(d[i] == 0):
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
                sources = deanonymize(G,pot,path,amt,pf.lnd_cost_fun)
                if sources != None:

                    #print("match",pot)
                    anon_sets[pot] = list(sources)
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

# def tr_at_adv(G,path,ads,amt,delay,ind,file):
#     delay -= G.edges[path[0],path[1]]["Delay"]
#     ind1 = -1
#     for i in range(1,len(path)-1):
#         #print(path[i],delay)
#         #delay = delay - G.edges[path[i], path[i + 1]]["Delay"]
#         #print(amt)
#         #amt = int((amt - G.edges[path[i], path[i + 1]]["BaseFee"]) / (1 + G.edges[path[i], path[i + 1]]["FeeRate"]))
#         if path[i] in ads:
#             #print("yes")
#             #adversary = path[i]
#             #ind1 = ads.index(path[i])
#             delay1 = delay-G.edges[path[i],path[i+1]]["Delay"]
#             amt1 = ((amt - G.edges[path[i], path[i + 1]]["BaseFee"]) / (1 + G.edges[path[i], path[i + 1]]["FeeRate"]))
#             B = dest_reveal_new(G,path[i],delay1,amt1,path[i-1],path[i+1])
#             #print(len(B))
#             #print("Adversary %d:",adversary,"Potential destinations are:")
#             #for j in range(0,len(B)):
#                 #print(B[j]["target"],B[j]["tech"],B[j]["sources"])
#             with open(file,'a') as csv_file:
#                 csvwriter = csv.writer(csv_file)
#                 for j in B:
#                     csvwriter.writerow([ind,path[i],j,B[j]])
#         delay -= G.edges[path[i], path[i + 1]]["Delay"]
#         amt = ((amt - G.edges[path[i], path[i + 1]]["BaseFee"]) / (1 + G.edges[path[i], path[i + 1]]["FeeRate"]))

def calc_params(G,path,amt):
    dist = 0
    cost = amt
    #print(dist,cost)
    for i in range(len(path)-2,-1,-1):
        #print(path[i],path[i+1])
        #print(G.edges[path[i],path[i+1]]["FeeRate"])
        fee = cost * G.edges[path[i], path[i+1]]["FeeRate"] + G.edges[path[i], path[i+1]]["BaseFee"]
        cost += fee
        dist += cost*G.edges[path[i],path[i+1]]["Delay"]*0.000000015 + fee
        #print(dist)
    return dist



def deanonymize(G,target,path,amt,cost_function):
    # if(target == 500):
    #     print("target", 6946,path)
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
        costs[node] = max
        paths[node] = []
        dists[node] = max
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
        if target == 500 and (curr == pre):
            print(curr)
        for [v,curr] in G.in_edges(curr):
            # if v == pre and curr == adv:
            #     print("yes1", pre,curr,G.edges[v, curr]["Balance"] + G.edges[curr, v]["Balance"],costs[curr])
            if (G.edges[v, curr]["Balance"] + G.edges[curr, v]["Balance"] >= costs[curr]) and v not in visited:
                # if v==pre:
                #     print("yes",pre)
                if done[v] == 0:
                    paths1[v] = [v]+paths[curr]
                    done[v] = 1
                cost = dists[curr]+ cost_function(G,costs[curr],curr,v)
                if cost < dists[v]:
                    paths[v] = [v]+paths[curr]
                    # if v==pre:
                    #     print(v,paths[v])
                    dists[v] = cost
                    delays[v] = delays[curr] + G.edges[v,curr]["Delay"]
                    costs[v] = costs[curr] + G.edges[v, curr]["BaseFee"] + costs[curr] * G.edges[v, curr]["FeeRate"]
                    # prob[v] = pf.edge_prob(G.edges[v,curr]["LastFailure"])*prob[curr]
                    pq.put((dists[v],v))
        if(curr in path[1:]):
            ind = path.index(curr)
            if(paths[curr]!=path[ind:]):
                return None
            if curr == adv:
                #print("ad", paths[curr])
                flag1 = 1
        if(curr == pre):
            # print(pre,paths[pre])
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




