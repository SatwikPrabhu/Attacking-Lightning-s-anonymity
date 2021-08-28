from queue import  PriorityQueue
from routingalgos.base import Routing
import nested_dict as nd
from math import inf
import random as rn

class PathSegmentRouting(Routing):

    # Initialize routing algorithm
    def __init__(self, baseRouting) -> None:
        super().__init__()
        self.__baseRouting = baseRouting

    # human readable name for this routing algorithm
    def name(self):
        return "Path Segment + " + self.__baseRouting.name()

    # Returns tech used by base routing  
    def tech(self):
        return self.__baseRouting.tech()

    # cost function, uses same cost function as base routing algorithm
    def cost_function(self, G, amount, u, v):
        return self.__baseRouting.cost_function(G, amount, u, v)


    # cost function for first hop, uses same cost function as base routing algorithm
    def cost_function_no_fees(self, G, amount, u, v):
        return self.__baseRouting.cost_function_no_fees(G, amount, u, v)


    def routePath(self, G, source, dest, amt):
        # check optimal path
        optroute = self.__baseRouting.routePath(G, source, dest, amt)
        optpath = optroute["path"]
        optdelay = optroute["delay"]
        optamount = optroute["amount"]
        optdist = optroute["dist"]

        # if nodes are directly connected, return immediately
        if (len(optpath) == 2):
            return {"dove":source, "path": optpath, "delay": optdelay, "amount": optamount, "dist": optdist}

        # choose best dovetail based on fewest hops
        best = 100
        bestpath = []
        bestdelay = -1
        bestamount = -1
        bestdist = -1
        bestdovetail = -1
        # try 5 dovetail candidates
        for i in range(5):
            dovetail = source
            while (dovetail in optpath):
                dovetail = rn.choice(list(G.nodes))
            path, delay, amount, dist = self.route_with_dove(G, source, dovetail, dest, amt)
            # pick candidate with shortest route & no loops
            if (len(path)> 0 and len(path) < best and len(path) == len(set(path))):
                best = len(path)
                bestpath = path
                bestdelay = delay
                bestamount = amount
                bestdist = dist
                bestdovetail = dovetail
        return {"dove": bestdovetail, "path": bestpath, "delay": bestdelay, "amount": bestamount, "dist": bestdist}

    def route_with_dove(self, G, source, dove, dest, amt):

        # route second path segment
        route2 = self.__baseRouting.routePath(G, dove, dest, amt, False)
        path2 = route2["path"]
        delay2 = route2["delay"]
        amount2 = route2["amount"]
        dist2 = route2["dist"]

        # return if infeasible
        if (len(path2) == 0):
            return [],-1,-1,-1
        
        # route first path segment, using new delay and amount
        route1 = self.__baseRouting.routePath(G, source, dove, amount2, True, delay2)
        path1 = route1["path"]
        delay1 = route1["delay"]
        amount1 = route1["amount"]
        dist1 = route1["dist"]

        # return if infeasible
        if (len(path1) == 0):
            return [],-1,-1,-1

        # append paths
        fullpath = path1 + path2[1:]
        fulldelay = delay1
        fullamount = amount1 
        fulldist = dist1 + dist2
        return fullpath, fulldelay, fullamount, fulldist
 

    def adversarial_attack(self, G,adversary,delay,amount,pre,next, attack_position):
        T = nd.nested_dict()

        flag1 = True
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
        
        if (True):
            # attack_position = 1
            return self.phase2_attack_position_known(level, T, G, amount, delay, pre, adversary, attack_position), flag1
        else:
            pass 

    def phase2_attack_position_known(self, level, T, G, amount, delay, pre, adversary, attack_position):
        anon_sets = nd.nested_dict()
        while(level>=0):
            t = T[level]["nodes"]
            d = T[level]["delays"]
            p = T[level]["previous"]
            a = T[level]["amounts"]
            v = T[level]["visited"]
            for i in range(0, len(t)):
                # shadow routing when in first path segment
                if(attack_position ==0 and d[i] >= 0 or attack_position > 0 and d[i] == 0):
                    path = []
                    level1 = level
                    path.append(T[level1]["nodes"][i])
                    loc = T[level1]["previous"][i]
                    while (level1 > 0):
                        level1 = level1 - 1
                        path.append(T[level1]["nodes"][loc])
                        loc = T[level1]["previous"][loc]
                    path.reverse()
                    # if attacker is the dovetail, look for source where the adversary is the target
                    if (attack_position == 1):
                        pot = path[-1]
                        advamt = amount + self.cost_function(G, amount, adversary, path[0])
                        advdel = delay + G.edges[adversary,path[0]]["Delay"]
                        sources = self.deanonymize(G, adversary, [pre, adversary], advamt, advdel)
                        if sources != None:
                            anon_sets[pot] = list(sources)
                    else:
                        path = [pre,adversary]+path
                        if (len(path) == len(set(path))):
                            amt = a[i]
                            dl = d[i]
                            pot = path[len(path) - 1]
                            sources = self.deanonymize(G,pot,path,amt,dl)
                            if sources != None:
                                anon_sets[pot] = list(sources)
            level = level - 1
        return anon_sets

    def deanonymize(self, G,target,path,amt,dl):
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
        # nxt = path[2]
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
        delays[target] = dl
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
                    cost = dists[curr] + self.cost_function(G,costs[curr],curr,v)
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