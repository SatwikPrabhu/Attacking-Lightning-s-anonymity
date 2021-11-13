from queue import PriorityQueue
from math import inf
import nested_dict as nd


class Routing:
    def __init__(self, ignore_tech = True):
        self.ignore_tech = ignore_tech
        self.collab = False

    # # cost function, should be overwritten by child
    # def cost_function(self, G, amount, u, v):
    #     print("error")
    #     pass

    # # cost function for first hop: sender does not take a fee
    # # should be overwritten by child
    # def cost_function_no_fees(self, G, amount, dist, u, v):
    #     print("error")
    #     pass

    # def route(self, G):
    #     print("error")
    #     pass

    # Dijkstra's routing algorithm for finding the shortest path
    def Dijkstra(self, G, source, target, amt, payment_source=True, target_delay=0):
        paths = {}
        dist = {}
        delay = {}
        amount = {}
        # prob = {}
        for node in G.nodes():
            amount[node] = -1
            delay[node] = -1
            dist[node] = inf    
        visited = set()
        pq = PriorityQueue()
        dist[target] = 0
        delay[target] = target_delay
        paths[target] = [target]
        amount[target] = amt
        pq.put((dist[target], target))
        while 0 != pq.qsize():
            curr_cost, curr = pq.get()
            if curr == source:
                return paths[curr], delay[curr], amount[curr], dist[curr]
            if curr_cost > dist[curr]:
                continue
            visited.add(curr)
            for [v, curr] in G.in_edges(curr):
                if payment_source and v == source and G.edges[v, curr]["Balance"] >= amount[curr]:
                    cost = dist[curr] + self.cost_function_no_fees(G, amount[curr], curr, v)
                    if cost < dist[v]:
                        dist[v] = cost
                        paths[v] = [v] + paths[curr]
                        delay[v] = G.edges[v, curr]["Delay"] + delay[curr]
                        amount[v] = amount[curr]
                        pq.put((dist[v], v))
                if(G.edges[v, curr]["Balance"] + G.edges[curr, v]["Balance"] >= amount[curr]) and v not in visited:
                    if (v != source or not payment_source):
                        cost = dist[curr] + self.cost_function(G, amount[curr], curr, v)
                        if cost < dist[v]:
                            dist[v] = cost
                            paths[v] = [v] + paths[curr]
                            delay[v] = G.edges[v, curr]["Delay"] + delay[curr]
                            amount[v] = amount[curr] + G.edges[v, curr]["BaseFee"] + \
                                amount[curr]*G.edges[v, curr]["FeeRate"]
                            pq.put((dist[v], v))
        return [], -1, -1, -1


    # adversarial attack
    def adversarial_attack(self, G,adversary,delay,amount,pre,next, attack_position = -1, shadow_routing = False):
        T = nd.nested_dict()
        anon_sets = nd.nested_dict()
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
        while(level>=0):
            t = T[level]["nodes"]
            d = T[level]["delays"]
            p = T[level]["previous"]
            a = T[level]["amounts"]
            v = T[level]["visited"]
            for i in range(0, len(t)):
                if d[i] == 0 or (shadow_routing and d[i] >= 0):
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
                        dl = d[i]
                        pot = path[len(path) - 1]
                        sources = self.deanonymize(G,pot,path,amt,dl)
                        if sources != None and len(sources) > 0:
                            anon_sets[pot] = list(sources)
            level = level - 1
        return anon_sets, flag1

    # find sources for a given target path
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
                    if done[v] == 0 and (G.nodes[v]["Tech"] == self.tech() or self.ignore_tech):
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
                    return None # TODO double check vs return []
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
                            if v not in paths[curr] and (G.nodes[v]["Tech"] == self.tech() or self.ignore_tech):
                                sources.append(v)
        sources = set(sources)
        return sources