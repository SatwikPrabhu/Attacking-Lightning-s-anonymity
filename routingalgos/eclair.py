from queue import PriorityQueue
from routingalgos.base import Routing
import nested_dict as nd
from math import inf
import random as rn
import requests


# Retrieves current block height from API
# in case of fail, will return a default block height
def getBlockHeight():
    print("Getting block height for Eclair...")
    API_URL = "https://api.blockcypher.com/v1/btc/main"
    try:
        CBR = requests.get(API_URL).json()['height']
        print("Block Height used:", CBR)
        return CBR
    except:
        print("Block height not found, using default 697000")
        return 697000


class EclairRouting(Routing):
    CBR = getBlockHeight()

    MIN_DELAY = 9
    MAX_DELAY = 2016
    MIN_CAP = 1
    MAX_CAP = 100000000
    MIN_AGE = 505149
    MAX_AGE = CBR #update to current block
    DELAY_RATIO = 0.15
    CAPACITY_RATIO = 0.5
    AGE_RATIO = 0.35

    # Initialize routing algorithm
    def __init__(self, ignore_tech = True) -> None:
        super().__init__(ignore_tech)

    # human-readable name for routing algorithm
    def name(self):
        return "Eclair"

    # tech label for this routing algorithm
    def tech(self):
        return 2

    def cost_function(self, G, amount, u, v):
        # if direct_conn:
        #     return 0
        fee = G.edges[v,u]['BaseFee'] + amount * G.edges[v,u]['FeeRate']
        ndelay = self.normalize(G.edges[v, u]["Delay"], self.MIN_DELAY, self.MAX_DELAY)
        ncapacity = 1 - (self.normalize((G.edges[v, u]["Balance"] + G.edges[u, v]["Balance"]), self.MIN_CAP, self.MAX_CAP))
        nage = self.normalize(self.CBR - G.edges[v, u]["Age"], self.MIN_AGE, self.MAX_AGE)
        alt = fee * (ndelay * self.DELAY_RATIO + ncapacity * self.CAPACITY_RATIO + nage * self.AGE_RATIO)
        return alt

    # cost function for first hop: sender does not take a fee
    def cost_function_no_fees(self, G, amount, u, v):
        return 0

    # construct route using Eclair algorithm, uses a modified general Dijkstra's algorithm
    def routePath(self, G, u, v, amt, payment_source=True, target_delay = 0 ):
        paths =  self.Dijkstra_general(G, u, v, amt, payment_source, target_delay)

        # fail when no paths found
        if(paths[0]==[]):
            return {"path": [], "delay": -1, "amount": -1, "dist": -1}

        # cut short when optimal path is direct
        if len(paths[0]) == 2:
            return {"path": paths[0], "delay": 0, "amount": amt, "dist": 0}
            
        # else, pick best of 3
        r = rn.randint(0, 2)
        path = paths[r]
        delay = target_delay
        amount = amt
        dist = 0
        # recalculate based on chosen path
        for m in range(len(path) - 2, 0, -1):
            delay += G.edges[path[m], path[m + 1]]["Delay"]
            amount += G.edges[path[m], path[m + 1]]["BaseFee"] + amount * G.edges[path[m], path[m + 1]]["FeeRate"]
        delay += G.edges[path[0], path[1]]["Delay"]
        return {"path": path, "delay": delay, "amount": amount, "dist": dist}

    # Generalized Dijkstra for 3 best paths - alternative to yen's algo
    def Dijkstra_general(self,G,source,target,amt, payment_source, target_delay):
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
        delay[target] = target_delay
        delay1[target] = target_delay
        delay2[target] = target_delay
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
                if payment_source and v == source and G.edges[v, curr]["Balance"] >= a:
                    #return [v] + paths[curr], delay[curr] + G.edges[v, curr]["Delay"], amount[curr], dist[curr]
                    path[k]= [v] + p
                    k+=1
                    if k == 3:
                        # print(path)
                        return path
                if (G.edges[v, curr]["Balance"] + G.edges[curr, v]["Balance"] >= a) and visited[v]<3 and v not in p:
                    if (v != source or not payment_source):
                        cost = di + self.cost_function(G, a, curr, v)

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

    # normalize between max and min
    def normalize(self, val, min, max):
        if val <= min:
                return 0.00001
        if val > max:
                return 0.99999
        return (val - min) / (max - min)


    # Returns potential sources that would use lnd to reach target using the subpath found. 
    def deanonymize(self, G,target,path,amt, dl):
        paths = {}
        paths1 = {}
        paths2 = {}
        path0 = {}
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
        pre = path[0]
        adv = path[1]
        nxt = path[2]
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
        delay[target] = dl
        delay1[target] = dl
        delay2[target] = dl
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
                if done[v] == 0 and (self.ignore_tech or G.nodes[v]["Tech"] == 2):
                    path0[v] = [v] + p
                    done[v] = 1
                elif done[v] == 1 and (self.ignore_tech or G.nodes[v]["Tech"] == 2):
                    path1[v] = [v] + p
                    done[v] = 2
                elif done[v] == 2 and (self.ignore_tech or G.nodes[v]["Tech"] == 2):
                    path2[v] = [v] + p
                    done[v] = 3
                if (G.edges[v, curr]["Balance"] + G.edges[curr, v]["Balance"] >= a) and visited[v] < 3 and v not in p:
                    cost = di + self.cost_function(G, a, curr, v)

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
            if(curr in path[1:]):
                ind = path.index(curr)
                if visited[curr] == 3:
                    if (paths[curr] != path[ind:] and paths1[curr] != path[ind:] and paths2[curr] != path[ind:]):
                        return None
                # If we find a match, then we do not need to look for other paths from curr
                if path[ind:] == p:
                    visited[curr] = 3
                    if curr == adv:
                        # print(paths[curr],paths1[curr],paths2[curr])
                        flag1 = 1
            if(curr == pre):
                if path == p or visited[curr] == 3:
                    visited[curr] = 3
                    # If pre is not an intermediary, then it must be the source
                if paths[pre] != path and paths1[pre]!=path and paths2[pre]!=path:
                    if (not self.ignore_tech and G.nodes[pre]["Tech"] != 2):
                        return None
                    return [pre]
                else:
                    # pre could still be a source
                    if (self.ignore_tech or G.nodes[pre]["Tech"] == 2):
                        sources.append(pre)
                    flag2 = 1
                # print(paths[curr], paths1[curr], paths2[curr])
            if flag1 == 1 and flag2 == 1:
                    # fill remaining possible sources
                    if pre in p:
                        for [v, curr] in G.in_edges(curr):
                            if v not in p and (self.ignore_tech or G.nodes[v]["Tech"] == 2):
                                sources.append(v)
    #                     ind = p.index(pre)
    #                     if p[ind:] == pa:
    #                         visited[curr] = 3
    #                         for [v, curr] in G.in_edges(curr):
    #                             if v not in paths[curr] and G.nodes[v]["Tech"] == 2:
    #                                 sources.append(v)
    #                     else:
    #                         print("error")
        sources = set(sources)
        return sources
   


    ######################## other routing functions (unused) ###################################

    def calc_params(self, G, path, amt):
        cost = amt
        delay = 0
        dist = 0
        for i in range(len(path)-2,0,-1):
            fee = cost * G.edges[path[i], path[i+1]]["FeeRate"] + G.edges[path[i], path[i+1]]["BaseFee"]

            ndelay = self.normalize(G.edges[path[i], path[i+1]]["Delay"], self.MIN_DELAY, self.MAX_DELAY)
            ncapacity = 1 - self.normalize((G.edges[path[i], path[i+1]]["Balance"] + G.edges[path[i], path[i+1]]["Balance"]), self.MIN_CAP, self.MAX_CAP)
            nage = self.normalize(self.CBR - G.edges[path[i], path[i+1]]["Age"], self.MIN_AGE, self.MAX_AGE)

            dist += fee * (ndelay * self.DELAY_RATIO + ncapacity * self.CAPACITY_RATIO + nage * self.AGE_RATIO)
            delay += G.edges[path[i],path[i+1]]["Delay"]
            cost += fee
        return dist

    # Original eclair implementation using yen's algorithm
    def Eclair(self, G, source, target, amt, path=None):
        G1 = G.copy()
        B = nd.nested_dict()
        if (path == None):
            B[0],d,c,di = self.routePath(G,source,target,amt)
            #print(B[0],di)
        # print(B[0]["Path"])
        else:
            B[0] = path
        paths = nd.nested_dict()
        leng = 0
        paths[leng]["Path"] = B[0]
        paths[leng]["Dist"] = self.calc_params(G, B[0], amt)
        paths[leng]["visited"] = 1
        leng += 1
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
                spurpath, delay, cost, dist = self.routePath(G1, spurnode, target, amt)
                totalpath = rootpath + spurpath[1:]

                flag = 0
                if totalpath == rootpath:
                    flag = 1
                for t in range(0, leng):
                    if (totalpath == paths[t]["Path"]):
                        flag = 1
                if flag == 0:
                    paths[leng]["Path"] = totalpath
                    di = self.calc_params(G, totalpath, amt)
                    paths[leng]["visited"] = 0
                    paths[leng]["Dist"] = di
                    #print(totalpath)
                    leng += 1
                for e in edges_removed:
                    u, v = e
                    G1.add_edge(u, v)
                    G1.edges[u, v]["Delay"] = G.edges[u, v]["Delay"]
                    G1.edges[u, v]["Balance"] = G.edges[u, v]["Balance"]
                    G1.edges[u, v]["BaseFee"] = G.edges[u, v]["BaseFee"]
                    G1.edges[u, v]["FeeRate"] = G.edges[u, v]["FeeRate"]
                    G1.edges[u, v]["Age"] = G.edges[u, v]["Age"]
            minpath = paths[1]["Path"]
            mincost = paths[1]["Dist"]
            index = -1
            for i in range(2, leng):
                if mincost > paths[i]["Dist"] and paths[i]["visited"] == 0:
                    mincost = paths[i]["Dist"]
                    minpath = paths[i]["Path"]
                    index = i
            paths[index]["visited"] = 1
            B[k] = minpath
        return B


    # modifying eclair's implementation to apply yen's algorithm with the spurnode moving from the destination to the source. Eclair originally does it the other way around.
    def modifiedEclair(self, G, source, target, amt, path=None):
        G1 = G.copy()
        B = nd.nested_dict()
        if (path == None):
            B[0],d,c,di = self.routePath(G,source,target,amt)
            #print(B[0],di)
        # print(B[0]["Path"])
        else:
            B[0] = path
        paths = nd.nested_dict()
        leng = 0
        paths[leng]["Path"] = B[0]
        paths[leng]["Dist"] = self.calc_params(G, B[0], amt)
        paths[leng]["visited"] = 1
        leng += 1
        for k in range(1, 3):
            A = B[k - 1]
            for i in range(len(A)-1, 0,-1):
                edges_removed = []
                spurnode = A[i]
                amt_spur = amt
                for j in range(len(A)-2,i-1,-1):
                    amt_spur = amt_spur + G.edges[A[j],A[j+1]]["BaseFee"] + amt_spur*G.edges[A[j],A[j+1]]["FeeRate"] 
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
                spurpath, delay, cost, dist = self.routePath(G1, source,spurnode, amt_spur)
                totalpath = spurpath[:-1]+rootpath

                flag = 0
                if totalpath == rootpath:
                    flag = 1
                for t in range(0, leng):
                    if (totalpath == paths[t]["Path"]):
                        flag = 1
                if flag == 0:
                    paths[leng]["Path"] = totalpath
                    di = self.calc_params(G, totalpath, amt)
                    paths[leng]["visited"] = 0
                    paths[leng]["Dist"] = di
                    #print(totalpath)
                    leng += 1
                if flag == 0:
                    # paths[leng]["Path"] = totalpath
                    di = self.calc_params(G, totalpath, amt)
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
            minpath = paths[1]["Path"]
            mincost = paths[1]["Dist"]
            index = -1
            for i in range(2, leng):
                if mincost > paths[i]["Dist"] and paths[i]["visited"] == 0:
                    mincost = paths[i]["Dist"]
                    minpath = paths[i]["Path"]
                    index = i
            paths[index]["visited"] = 1
            B[k] = minpath
        return B

