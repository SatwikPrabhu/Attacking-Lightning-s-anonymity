from queue import  PriorityQueue
from routingalgos.base import Routing
import nested_dict as nd
from math import inf
import random as rn
# import time

class PathSegmentRouting(Routing):

    # Initialize routing algorithm
    def __init__(self, base_routing, position_known = False, collab = False) -> None:
        super().__init__()
        self.__base_routing = base_routing
        self.__position_known = position_known
        self.collab = collab

    # human readable name for this routing algorithm
    def name(self):
        return "Path Segment + " + self.__base_routing.name()

    # Returns tech used by base routing  
    def tech(self):
        return self.__base_routing.tech()

    # cost function, uses same cost function as base routing algorithm
    def cost_function(self, G, amount, u, v):
        return self.__base_routing.cost_function(G, amount, u, v)


    # cost function for first hop, uses same cost function as base routing algorithm
    def cost_function_no_fees(self, G, amount, u, v):
        return self.__base_routing.cost_function_no_fees(G, amount, u, v)


    def routePath(self, G, source, dest, amt):
        # check optimal path
        optroute = self.__base_routing.routePath(G, source, dest, amt)
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
        # for i in range(5):
        #     dovetail = source
        #     while (dovetail in optpath):
        #         dovetail = rn.choice(list(G.nodes))
        #     path, delay, amount, dist = self.route_with_dove(G, source, dovetail, dest, amt)
        #     # pick candidate with shortest route & no loops
        #     if (len(path)> 0 and len(path) < best and len(path) == len(set(path))):
        #         best = len(path)
        #         bestpath = path
        #         bestdelay = delay
        #         bestamount = amount
        #         bestdist = dist
        #         bestdovetail = dovetail

        # Just pick a random node as the dovetail
        dovetail = source
        while (dovetail in [source, dest]):
            dovetail = rn.choice(list(G.nodes))
        bestdovetail = dovetail
        bestpath, bestdelay, bestamount, bestdist = self.route_with_dove(G, source, dovetail, dest, amt)
        return {"dove": bestdovetail, "path": bestpath, "delay": bestdelay, "amount": bestamount, "dist": bestdist}

    def route_with_dove(self, G, source, dove, dest, amt):

        # route second path segment
        route2 = self.__base_routing.routePath(G, dove, dest, amt, False)
        path2 = route2["path"]
        delay2 = route2["delay"]
        amount2 = route2["amount"]
        dist2 = route2["dist"]

        # return if infeasible
        if (len(path2) == 0):
            return [],-1,-1,-1
        
        # route first path segment, using new delay and amount
        route1 = self.__base_routing.routePath(G, source, dove, amount2, True, delay2)
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
 
    def adversarial_attack(self, G,adversary,delay,amount,pre,next, attack_position = -1):
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
        
        if (self.__position_known and attack_position != -1):
            return self.phase2_attack_position_known(level, T, G, amount, delay, pre, adversary, attack_position), flag1
        else:
            return self.phase2_attack_position_unknown(level, T, G, amount, delay, pre, adversary), flag1

    # looks for all possible sources when assuming a certain attack position
    def phase2_attack_position_known(self, level, T, G, amount, delay, pre, adversary, attack_position):
        anon_sets = {}
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
                        if sources != None and len(sources) > 0:
                            anon_sets[pot] = list(sources)
                    else:
                        path = [pre,adversary]+path
                        if (len(path) == len(set(path))):
                            amt = a[i]
                            dl = d[i]
                            pot = path[len(path) - 1]
                            sources = self.deanonymize(G,pot,path,amt,dl)
                            if sources != None and len(sources) > 0:
                                anon_sets[pot] = list(sources)
            level = level - 1
        return anon_sets

    # looks for all possible source/dest pairs for every possible attack position, and combines results
    def phase2_attack_position_unknown(self, level, T, G, amount, delay, pre, adversary):
        first_sources = self.phase2_attack_position_known(level, T, G, amount, delay, pre, adversary, 0)
        # center_sources = self.phase2_attack_position_known(level, T, G, amount, delay, pre, adversary, 1)
        # second_sources = self.phase2_attack_position_known(level, T, G, amount, delay, pre, adversary, 2)

        # for dst in center_sources:
        #     if dst in first_sources:
        #         first_sources[dst] = list(set(first_sources[dst] + center_sources[dst]))
        # for dst in second_sources:
        #     if dst in first_sources:
        #         first_sources[dst] = list(set(first_sources[dst] + second_sources[dst]))

        return first_sources

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

    
    # Attack that combines the information of multiple attackers,
    # known as 'witnesses'.
    # This attack assumes that the transactions witnessed by all
    # witnesses is the same transaction with 100% certainty.
    def collusion_attack(self, G, witnesses):

        # collusion attack cannot take place if there is only 1 witness
        if len(witnesses) < 2:
            return None, False
        
        # Pick the outer most witnesses of the transaction
        # (closest to src and dest)
        first_adv = witnesses[0]
        last_adv = witnesses[-1]

        # print(first_adv)
        # print(last_adv)

        # route between earliest and last
        n1 = first_adv["prev"]
        n2 = last_adv["nxt"]

        target_amt = last_adv["amt"]
        target_delay = last_adv["delay"]

        # use base to find optimal route
        optpath = self.__base_routing.routePath(G, n1, n2, target_amt, target_delay=target_delay)

        time_delta = optpath["delay"] - target_delay

        found_delay = first_adv["delay"] + G.edges[first_adv["n"], first_adv["nxt"]]["Delay"] + G.edges[n1, first_adv["n"]]["Delay"]
        found_delta = found_delay - target_delay


        if (found_delta == time_delta):
            # optimal path it seems
            # collaboration attack not possible, return None
            return None, False
        else:
            # ads are on different path segments, or node before first ad is src
            # here we only take the first option into account
            
            u = first_adv["nxt"]
            v = last_adv["prev"]
            u_timelock = first_adv["delay"]
            v_timelock = last_adv["delay"] + G.edges[v, last_adv["n"]]["Delay"] + G.edges[last_adv["n"], n2]["Delay"]
            u_amt = first_adv["amt"]
            v_amt = last_adv["amt"]
            v_amt = v_amt + G.edges[last_adv["n"], n2]["BaseFee"] + v_amt * G.edges[last_adv["n"], n2]["FeeRate"]
            v_amt = v_amt + G.edges[v, last_adv["n"]]["BaseFee"] + v_amt * G.edges[v, last_adv["n"]]["FeeRate"]

            advpath = self.find_path_new(G, u, v, u_timelock, v_timelock, u_amt, v_amt)

            # print("advpath", advpath)

            if len(advpath) > 0:
                advpath = [ n1, first_adv["n"]] + advpath
                return self.coll_adv_attack(G, last_adv["n"], last_adv["delay"], last_adv["amt"], last_adv["prev"], last_adv["nxt"], advpath)
            else:
                # print("advpath not found")
                return None, False

    # Tries to find the path between two nodes with known timelocks and amounts.
    def find_path_new(self, G, start, goal, tstart, tgoal, amtstart, amtgoal):
        pq = PriorityQueue()
        pq.put((tgoal, (goal, -1, amtgoal, amtgoal)))
        nxts = {}

        while not pq.empty():
            t, (cur, nxt, amt, amtold) = pq.get()
            nxts[(cur, t, amt)] = (nxt, amtold)

            if (t > tstart):
                return []
            
            if cur == start and t == tstart and abs(amtstart - amt) < 0.001:
                path = [cur]
                while nxts[(cur, t, amt)][0] != -1:
                    tnew = t - G.edges[cur, nxts[(cur, t, amt)][0]]["Delay"]
                    amtnew = nxts[(cur,t,amt)][1]
                    cur = nxts[(cur, t, amt)][0]
                    t = tnew
                    amt = amtnew
                    path.append(cur)
                return path

            for [p, cur] in G.in_edges(cur):
                if (p != nxt):
                    tnew = t + G.edges[p, cur]["Delay"]
                    amtnew = amt + G.edges[p, cur]["BaseFee"] + amt * G.edges[p, cur]["FeeRate"]
                    if amtnew < amtstart + 0.001:
                        pq.put((tnew, (p, cur, amtnew, amt)))
        return []

    # Tries to find the path between two nodes using the known timelocks.
    # Not used anymore, replaced with function that also looks at amount.
    def find_path_old(self, G, start, goal, tstart, tgoal):
        pq = PriorityQueue()
        pq.put((-tstart, (start,-1)))
        prevs = {}

        while not pq.empty():
            t, (c, p) = pq.get()
            t = -t

            prevs[(c, t)] = p

            if (t < tgoal):
                return []

            if c == goal and t == tgoal:
                path = [c]
                while prevs[(c, t)] != -1:
                    tnew = t + G.edges[prevs[(c,t)], c]["Delay"]
                    c = prevs[(c,t)]
                    t = tnew
                    path.append(c)
                path.reverse()
                return path

        for [c, v] in G.out_edges(c):
            if (v != p):
                pq.put(( -(t - G.edges[c,v]["Delay"]), (v, c)))
        return []

    # Function that does the collaboration attack.
    # Uses the found adversarial path.
    # Does a phase 1 attack from the last node, tries to find the earlist optimal node
    # (dovetail), then does a phase 2 attack with the found dovetail node as the target.
    def coll_adv_attack(self, G, adversary, delay, amount, pre, next, advpath):
                # tp1_begin = time.time()
                T = nd.nested_dict()
                anon_sets = {}
                flag1 = True
                level = 0
                T[0]["nodes"] = [next]
                T[0]["delays"] = [delay]
                T[0]["previous"] = [-1]
                T[0]["visited"] = [[pre,adversary,next]]
                T[0]["amounts"] = [amount]
                flag = True

                # cache sources for all candidate dovetail nodes (they are identical)
                sourcesets = {}

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
                        if d[i] == 0:
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

                                # find earliest possible dovetail node for this path
                                dove, dove_index = self.get_dovetail(G, advpath, path, amt)
                                if dove_index != -1:
                                    if dove not in sourcesets:
                                        fullpath = advpath + path[1:]

                                        ind = len(fullpath) - 1
                                        while ind > dove_index:
                                            amt += G.edges[fullpath[ind-1], fullpath[ind]]["BaseFee"] + amt * G.edges[fullpath[ind-1], fullpath[ind]]["FeeRate"]
                                            dl += G.edges[fullpath[ind-1], fullpath[ind]]["Delay"]
                                            ind -= 1

                                        # tp2_begin = time.time()
                                        sources = self.deanonymize(G,dove,fullpath[:dove_index+1],amt,dl)
                                        # tp2_end = time.time()
                                        # print("Time for candidate {}: {} seconds".format(pot, tp2_end - tp2_begin))
                                        sourcesets[dove] = sources
                                    else:
                                        # print("Source from cache of dovetail {}.".format(dove))
                                        sources = sourcesets[dove]
                                    if sources != None and len(sources) > 0:
                                        anon_sets[pot] = list(sources)
                                # else:
                                #     print("Dovetail not found for candidate {}".format(pot))
                    level = level - 1
                # tp1_end = time.time()
                # print("Time for full: {} seconds".format( tp1_end - tp1_begin))
                return anon_sets, flag1

    # Finds the earliest possible node from which the full path is found
    # to be optimal, and assume this is the dovetail node.
    def get_dovetail(self, G, advpath, endpath, amt):
        for i in range(len(advpath)-2):
            path = advpath[i+2:] + endpath[1:]
            if self.check_optimal(G, path, amt):
                # return dove and index
                return advpath[i+2], i+2
        return -1, -1

    # Find and compare the found route with  the optimal route
    # found using the base routing algorithm.
    def check_optimal(self, G, path, amt):
        res = self.__base_routing.routePath(G, path[0], path[-1], amt, payment_source = False )
        return path == res["path"]