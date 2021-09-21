from queue import  PriorityQueue
from routingalgos.base import Routing
from math import inf

class LNDRouting(Routing):
    LND_RISK_FACTOR = 0.000000015
    A_PRIORI_PROB = 0.6

    # Initialize routing algorithm
    def __init__(self) -> None:
        super().__init__()

    # human-readable name for routing algorithm
    def name(self):
        return "LND"

    # tech label for this routing algorithm
    def tech(self):
        return 0

    # cost function for lnd. We ignore the probability bias aspect for now
    def cost_function(self, G, amount, u, v):
        # if direct_conn:
        #     return amount[v] * G.edges[v, u]["Delay"] * LND_RISK_FACTOR
        fee = G.edges[v,u]['BaseFee'] + amount * G.edges[v, u]['FeeRate']
        alt = (amount+fee) * G.edges[v, u]["Delay"] * self.LND_RISK_FACTOR + fee

        # t = G.edges[v, u]["LastFailure"]
        # edge_proba = edge_prob(t)
        # edge_proba *= prob
        # alt = prob_bias(alt,edge_proba)
        return alt

    # cost function for first hop: sender does not take a fee
    def cost_function_no_fees(self, G, amount, u, v):
        return amount*G.edges[v,u]["Delay"]*self.LND_RISK_FACTOR


    # construct route using lnd algorithm (uses ordinary dijkstra)
    def routePath(self, G, u, v, amt, payment_source=True, target_delay = 0 ):
        path, delay, amount, dist =  self.Dijkstra(G, u, v, amt, payment_source, target_delay)
        return {"path": path, "delay": delay, "amount": amount, "dist": dist}
