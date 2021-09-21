from routingalgos.randomhops import RandomHopsRouting
from routingalgos.eclair import EclairRouting
from routingalgos.clightning import CLightningRouting
from routingalgos.lnd import LNDRouting
from routingalgos.pathsegment import PathSegmentRouting
import networkx as nx
import json
import random as rn
import signal


def handler(signum, frame):
    x = input(
        "Ctrl-c was pressed. Want to print results to file? (enter to print, type anything to cancel) ")
    if (x == ""):
        print("Printing to File ... ")
        with open(file, 'r') as json_file:
            data = json.load(json_file)
            data.append(transactions)
        with open(file, 'w') as json_file:
            json.dump(data, json_file, indent=1)
    exit(1)


signal.signal(signal.SIGINT, handler)

NODES = 100
EDGES = 2
RANDOMNESS = 65
TRANSACTIONS = 1000

# LND routing
# routingAlgo = LNDRouting()

# C-lightning routing
# fuzz = rn.uniform(-1,1)
# routingAlgo = CLightningRouting(fuzz)

# Eclair routing
# routingAlgo = EclairRouting()

# Random Hop routing
# routingAlgo = RandomHopsRouting(LNDRouting(), False) # True = first attack strat, False = second

# dovetail routing (with given routing basis)
routingAlgo = PathSegmentRouting(LNDRouting(), False)


def simulate_tx(G, path, dove, delay, amt, ads, amt1, file):
    G1 = G.copy()
    cost = amt
    comp_attack = []
    anon_sets = {}
    attack_positions = {}
    attacked = 0
    G.edges[path[0], path[1]]["Balance"] -= amt
    G.edges[path[1], path[0]]["Locked"] = amt
    delay = delay - G.edges[path[0], path[1]]["Delay"]
    i = 1

    if (dove != -1):
        dove_connectivity = len(G.in_edges(dove))
    else:
        dove_connectivity = -1

    if len(path) == 2:
        G.edges[path[1], path[0]]["Balance"] += G.edges[path[1], path[0]]["Locked"]
        G.edges[path[1], path[0]]["Locked"] = 0
        transaction = {"sender": path[0], "recipient": path[1], "dovetail": dove,
                        "dove_connectivity": dove_connectivity, "path": path,
                        "attack_position": attack_positions, "delay": delay,
                        "amount": amt1, "Cost": cost, "attacked": 0, "success": True,
                        "anon_sets": anon_sets, "comp_attack": comp_attack}
        transactions.append(transaction)
        return True
    while(i < len(path)-1):
        amt = (amt - G.edges[path[i], path[i+1]]["BaseFee"]
               ) / (1 + G.edges[path[i], path[i+1]]["FeeRate"])
        if path[i] in ads:

            if (dove != -1):
                # find the attacker's real position
                # this is not necessarily used in the algorithm, but is used in results
                if (path.index(dove) < path.index(path[i])):
                    attack_position = 2
                if (path.index(dove) > path.index(path[i])):
                    attack_position = 0
                if (path.index(dove) == path.index(path[i])):
                    attack_position = 1
            else:
                attack_position = -1

            attacked += 1
            delay1 = delay - G.edges[path[i], path[i+1]]["Delay"]
            dests, flag = routingAlgo.adversarial_attack(
                G1, path[i], delay1, amt, path[i-1], path[i+1], attack_position)
            if flag:
                comp_attack.append(1)
            else:
                comp_attack.append(0)
            anon_sets[path[i]] = dests
            attack_positions[path[i]] = attack_position
        if(G.edges[path[i], path[i+1]]["Balance"] >= amt):
            G.edges[path[i], path[i+1]]["Balance"] -= amt
            G.edges[path[i+1], path[i]]["Locked"] = amt
            if i == len(path) - 2:
                G.edges[path[i+1], path[i]
                        ]["Balance"] += G.edges[path[i+1], path[i]]["Locked"]
                G.edges[path[i+1], path[i]]["Locked"] = 0
                j = i - 1
                while j >= 0:
                    G.edges[path[j + 1], path[j]
                            ]["Balance"] += G.edges[path[j + 1], path[j]]["Locked"]
                    G.edges[path[j + 1], path[j]]["Locked"] = 0
                    j = j-1
                transaction = {"sender": path[0], "recipient": path[len(path)-1], "dovetail": dove,
                                "dove_connectivity": dove_connectivity, "path": path,
                                "attack_position": attack_positions, "delay": delay, "amount": amt1,
                                "Cost": cost, "attacked": attacked, "success": True,
                                "anon_sets": anon_sets, "comp_attack": comp_attack}
                transactions.append(transaction)
                return True
            delay = delay - G.edges[path[i], path[i+1]]["Delay"]
            i += 1
        else:
            j = i - 1
            while j >= 0:
                G.edges[path[j], path[j+1]
                        ]["Balance"] += G.edges[path[j+1], path[j]]["Locked"]
                G.edges[path[j + 1], path[j]]["Locked"] = 0
                j = j-1
            transaction = {"sender": path[0], "recipient": path[len(path)-1], "dovetail": dove,
                            "dove_connectivity": dove_connectivity, "path": path,
                            "attack_position": attack_positions, "delay": delay, "amount": amt1,
                            "Cost": cost, "attacked": attacked, "success": False,
                            "anon_sets": anon_sets, "comp_attack": comp_attack}
            transactions.append(transaction)
            return False


G = nx.barabasi_albert_graph(NODES, EDGES, RANDOMNESS)
G = nx.DiGraph(G)

rn.seed(RANDOMNESS)

for [u, v] in G.edges():
    G.edges[u, v]["Delay"] = 10 * rn.randint(1, 10)
    G.edges[u, v]["BaseFee"] = 0.1 * rn.randint(1, 10)
    G.edges[u, v]["FeeRate"] = 0.0001 * rn.randint(1, 10)
    G.edges[u, v]["Balance"] = rn.randint(100, 10000)
    G.edges[u, v]["Age"] = 1000 * rn.randint(500, 600)

# every node uses the same tech
for u in G.nodes():
    G.nodes[u]["Tech"] = routingAlgo.tech()

transactions = []

B = nx.betweenness_centrality(G)

ads = []
for i in range(0, 10):
    node = -1
    max = -1
    for u in G.nodes():
        if B[u] >= max and u not in ads:
            max = B[u]
            node = u
    if node not in ads:
        ads.append(node)

print("Adversaries:", ads)

i = 0
file = "results.json"
while (i < TRANSACTIONS):
    u = -1
    v = -1
    while (u == v or (u not in G.nodes()) or (v not in G.nodes())):
        u = rn.randint(0, NODES - 1)
        v = rn.randint(0, NODES - 1)
    amt = 0
    if (i % 3 == 0):
        amt = rn.randint(1, 10)
    elif (i % 3 == 1):
        amt = rn.randint(10, 100)
    elif (i % 3 == 2):
        amt = rn.randint(100, 1000)
    route = routingAlgo.routePath(G, u, v, amt)

    path = route["path"]
    delay = route["delay"]
    amount = route["amount"]
    dist = route["dist"]

    if ("dove" in route):
        dove = route["dove"]
    else:
        dove = -1

    if (len(path) > 0):
        T = simulate_tx(G, path, dove, delay, amount, ads, amt, file)
    if len(path) > 2:
        print(i, path, "done")
        i += 1
with open(file, 'r') as json_file:
    data = json.load(json_file)
data.append(transactions)
with open(file, 'w') as json_file:
    json.dump(data, json_file, indent=1)
