from routingalgos.shadow_routing import ShadowRouting
from routingalgos.randomhops import RandomHopsRouting
from routingalgos.pathsegment import PathSegmentRouting
from routingalgos.eclair import EclairRouting
from routingalgos.clightning import CLightningRouting
from routingalgos.lnd import LNDRouting
import populate_graph as pg
import networkx as nx
import random as rn
import json
from mpi4py import MPI
import signal
import numpy as np
import os
import sys
import time

# Generate routing algorithm objects.
# Note: not all need to be used!
# See line 32 for currently used algorithms.
LNDRoutingObj = LNDRouting()
CLightningRoutingObj = CLightningRouting(rn.uniform(-1,1))
EclairRoutingObj = EclairRouting()
PathSegmentRoutingObj = PathSegmentRouting(LNDRouting(), False)
RandomHopsRoutingObj = RandomHopsRouting(LNDRouting(), True)
ShadowRoutingObj = ShadowRouting(LNDRouting())
CollabRoutingObj = PathSegmentRouting(LNDRouting(), False, collab = True)

# Algorithms that are to be executed with the current transaction set.
# These can be changed.
# Length of algorithms and aliases should be the same!
algorithms = [LNDRoutingObj, CLightningRoutingObj, EclairRoutingObj, RandomHopsRoutingObj]
aliases = ["lnd", "cln", "ecl", "rnd"]

# Interrupt handler, simply prints out ("Exiting program").
# Gets called when sending interrupt signal (CTRL-C).
def inthandler(signum, frame):
    print("Exiting program ...")
    # printtofiles()
    sys.stdout.flush()
    exit(0)

# Handler for printing current state of program.
# Useful for debugging purposes.
# This can be called by sending signal 10 to the running process.
# Example: pkill -10 python
def printhandler(signum, frame):
    print("--------------------------")
    print("Proc {}".format(rank))
    print("Txs: {}".format(len(transactions[-1])))
    print(log_location)
    print("Time for this action: {:.2f} seconds".format(time.time() - log_time ))
    print("--------------------------")
    sys.stdout.flush()

# Handler for doing a graceful termination of the program.
# When calling this handler, the program will terminate after
# the current transaction has finished.
# That is: all algorithms are finished with simulating this tx.
# This can be called hy sending signal 12 to the running process.
# Example: pkill -12 python
def graceful(signum, frame):
    global endflag
    print("Terminate when current tx is finished")
    endflag = True

# Helper function for printing all transactions to files.
# Can get called either when terminating program, or after
# every simulated transaction.
def printtofiles():
    for j in range(len(filenames)):
        with open(filenames[j], 'w+') as json_file:
            json.dump(transactions[j], json_file, indent=1)
    return

# Assign handlers to the correct signals.
# Note: when using mpi, these are some of the few signals
# that can be assigned to!
signal.signal(signal.SIGINT, inthandler)
signal.signal(10, printhandler)
signal.signal(12, graceful)

# Used for logging.
log_location = "Not started"
log_time = time.time()
endflag = False

# Variables used if MPI is used, to differentiate between the processes.
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Create output directory (if one does not exist already).
if not os.path.isdir("output"):
    try:
        os.mkdir("output")
    except:
        pass

# Filenames of files that store details of all
# transactions and all attack results,
# and list that stores all these results in memory.
filenames = []
transactions = []
for a in aliases:
    filenames.append("output/results-{}-{}.json".format(a,str(comm.Get_rank())))
    transactions.append([])



# Load transactions from file created by txcreate.py.
tx_list = np.loadtxt("txs.txt", dtype=int)
mpi_size = comm.Get_size()
tx_index = int(comm.Get_rank() * len(tx_list)/mpi_size)

# Populate the graph from the snapshot.
G = nx.DiGraph()
G,m = pg.populate_nodes(G)
G,m1=pg.populate_channels(G,m,645320)
G = pg.populate_policies(G,m1)

print("Graph imported")

# Curate nodes and channels removing channels that are closed and those that do not have public policies.
G1 = nx.DiGraph()
for [u,v] in G.edges():
    if(G.edges[u,v]["marked"]==1 and G.edges[v,u]["marked"]==1):
        if (u not in G1.nodes()):
            G1.add_node(u)
            G1.nodes[u]["name"] = G.nodes[u]["name"]
            G1.nodes[u]["pubadd"] = G.nodes[u]["pubadd"]
            G1.nodes[u]["Tech"] = G.nodes[u]["Tech"]
            #print(G1.nodes[u]["Tech"])
        if (v not in G1.nodes()):
            G1.add_node(v)
            G1.nodes[v]["name"] = G.nodes[v]["name"]
            G1.nodes[v]["pubadd"] = G.nodes[v]["pubadd"]
            G1.nodes[v]["Tech"] = G.nodes[v]["Tech"]
            #print(G1.nodes[v]["Tech"])
        G1.add_edge(u,v)
        G1.edges[u,v]["Balance"] = G.edges[u,v]["Balance"]
        G1.edges[u, v]["Age"] = G.edges[u, v]["Age"]
        G1.edges[u, v]["BaseFee"] = G.edges[u, v]["BaseFee"]
        G1.edges[u, v]["FeeRate"] = G.edges[u, v]["FeeRate"]
        G1.edges[u, v]["Delay"] = G.edges[u, v]["Delay"]
        G1.edges[u, v]["id"] = G.edges[u, v]["id"]

# Create a deep copy of the graph for every routing algorithm.
Gs = []
for _ in algorithms:
    Gs.append(G1.copy())

print("Graph filtered")

# Simulate the payment, try to de-anonymize if the adversary is encountered and fail if any of the balances are not sufficient        
def simulate_tx(G,path,dove, delay,amt,ads,amt1, routing_algo):
    global log_location, log_time

    tech = routing_algo.name()
    cost = amt
    comp_attack = []
    anon_sets = {}
    attacked = 0
    G1 = G.copy()
    G.edges[path[0],path[1]]["Balance"] -= amt
    G.edges[path[1],path[0]]["Locked"] = amt
    delay = delay - G.edges[path[0],path[1]]["Delay"]
    i = 1
    witnesses = []

    if len(path) == 2:
        G.edges[path[1],path[0]]["Balance"] += G.edges[path[1],path[0]]["Locked"]
        G.edges[path[1], path[0]]["Locked"] = 0
        transaction = {"sender": path[0], "recipient": path[1], "path" : path, "delay": delay,
                        "amount":amt1, "Cost": cost, "tech":tech, "attacked":0, "success":True,
                        "anon_sets":anon_sets,"comp_attack":comp_attack}
        if dove != -1:
            transaction["dovetail"] = dove
        return True, transaction
    while(i < len(path)-1):
        amt = (amt - G.edges[path[i], path[i+1]]["BaseFee"]) / (1 + G.edges[path[i], path[i+1]]["FeeRate"])
        if path[i] in ads:
            attacked+=1
            dests = []
            delay1 = delay - G.edges[path[i],path[i+1]]["Delay"]
            log_time = time.time()
            log_location = "{} routing, \n path: {}, \n current: {} (attacker), \n delay: {}, \n amt: {}".format(tech, path, path[i], delay1, amt)

            if not routing_algo.collab:
                dests, flag = routing_algo.adversarial_attack(G1, path[i], delay1, amt, path[i-1], path[i+1])
                if flag == True:
                    comp_attack.append(1)
                else:
                    comp_attack.append(0)
                anon_sets[path[i]] = dests
            else:
                witnesses.append({"n":path[i], "amt": amt, "delay": delay1, "prev": path[i-1], "nxt": path[i+1]})
        if(G.edges[path[i],path[i+1]]["Balance"] >= amt):
            log_location = "{} routing, path: {}, current: {} (index {})".format(tech, path, path[i], i)
            log_time = time.time()
            G.edges[path[i], path[i+1]]["Balance"] -= amt
            G.edges[path[i+1], path[i]]["Locked"] = amt
            if i == len(path) - 2:
                G.edges[path[i+1],path[i]]["Balance"] += G.edges[path[i+1], path[i]]["Locked"]
                G.edges[path[i+1], path[i]]["Locked"] = 0
                j = i - 1
                while j >= 0:
                    G.edges[path[j + 1], path[j]]["Balance"] += G.edges[path[j + 1], path[j]]["Locked"]
                    G.edges[path[j + 1], path[j]]["Locked"] = 0
                    j = j-1

                if routing_algo.collab:
                    log_time = time.time()
                    log_location = "{} routing, \n path: {}, \n  collaboration attack after succesful tx".format(tech, path)
                    dests, flag =  routing_algo.collusion_attack(G, witnesses)
                    if dests is not None:
                        for w in witnesses:
                            anon_sets[w["n"]] = dests
                transaction = {"sender": path[0], "recipient": path[len(path)-1], "path": path, "delay": delay,
                                "amount": amt1, "Cost": cost,"tech":tech, "attacked": attacked,
                                "success": True, "anon_sets": anon_sets, "comp_attack": comp_attack}
                if dove != -1:
                    transaction["dovetail"] = dove
                return True, transaction
            delay = delay - G.edges[path[i],path[i+1]]["Delay"]
            i += 1
        else:
            log_location = "{} routing, path: {}, current: {} (index {}), routing failed".format(tech, path, path[i], i)
            log_time = time.time()
            j = i - 1
            while j >= 0:
                G.edges[path[j],path[j+1]]["Balance"] += G.edges[path[j+1],path[j]]["Locked"]
                G.edges[path[j + 1], path[j]]["Locked"] = 0
                j = j-1
            if routing_algo.collab:
                log_location = "{} routing, \n path: {}, \n  collaboration attack after failed tx".format(tech, path)
                log_time = time.time()
                dests, flag =  routing_algo.collusion_attack(G, witnesses)
                if dests is not None:
                    for w in witnesses:
                        anon_sets[w["n"]] = dests
            transaction = {"sender": path[0], "recipient": path[len(path)-1], "path": path, "delay": delay,
                            "amount": amt1, "Cost": cost,"tech":tech, "attacked": attacked,
                            "success": False, "anon_sets": anon_sets, "comp_attack": comp_attack}
            if dove != -1:
                transaction["dovetail"] = dove
            return False, transaction

i = 0
# list of adversaries with a mix of nodes with high centrality, low centrality and random nodes.
# This can be changed as per requirement. Same goes for the number of transactions.
ads = [17760, 7947, 1128, 2828, 3066, 22248, 9012, 22397, 19494, 5114]
# ads = [2634, 8075, 5347, 1083, 5093,4326, 4126, 2836, 5361, 10572,5389, 3599, 9819, 4828, 3474, 8808, 93, 9530, 9515, 2163]

while(i<(len(tx_list) // mpi_size) and not endflag):
    u = int(tx_list[tx_index,0])
    v = int(tx_list[tx_index,1])
    amt = int(tx_list[tx_index,2])
    tx_index += 1
    cur = 0

    paths = []
    delays = []
    amounts = []
    doves = []

    for cur in range(len(algorithms)):
        log_location = "Pathfinding {}, u: {}, v: {}, amt: {}".format(aliases[cur], u,v,amt)
        log_time = time.time()
        routingObj = algorithms[cur]
        G = Gs[cur]
        route_result = routingObj.routePath(G, u, v, amt)
        paths.append(route_result["path"])
        delays.append(route_result["delay"])
        amounts.append(route_result["amount"])
        if "dove" in route_result:
            doves.append(route_result["dove"])
        else:
            doves.append(-1)

    all_success = True
    for l in paths:
        if len(l) == 0:
            all_success = False

    if all_success:
        for cur in range(len(algorithms)):
            log_location = "Routing {}".format(aliases[cur])
            log_time = time.time()
            routingObj = algorithms[cur]
            G = Gs[cur]
            path = paths[cur]
            dove = doves[cur]
            delay = delays[cur]
            amount = amounts[cur]
            T,new_tx  = simulate_tx(G, path, dove, delay, amount, ads, amt, routingObj)
            transactions[cur].append(new_tx)
        printtofiles()

    i += 1
