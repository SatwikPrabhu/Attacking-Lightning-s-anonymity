from queue import PriorityQueue
from routingalgos.base import Routing
import nested_dict as nd
from math import inf

class CLightningRouting(Routing):
    C_RISK_FACTOR = 10
    RISK_BIAS = 1
    DEFAULT_FUZZ = 0.05

    # Initialize routing algorithm
    def __init__(self, fuzz, ignore_tech = True) -> None:
        super().__init__(ignore_tech)
        self.__fuzz = fuzz

    # human-readable name for routing algorithm
    def name(self):
        return "C-Lightning"

    # tech label for this routing algorithm
    def tech(self):
        return 1

    def cost_function(self, G, amount, u, v):
        # if direct_conn:
        #     return amount[v] * G.edges[v, u]["Delay"] * C_RISK_FACTOR + RISK_BIAS
        scale = 1 + self.DEFAULT_FUZZ * self.__fuzz
        fee = scale * (G.edges[v, u]['BaseFee'] + amount * G.edges[v, u]['FeeRate'])
        alt = ((amount + fee) * G.edges[v, u]["Delay"]
               * self.C_RISK_FACTOR + self.RISK_BIAS)
        return alt

    # cost function for first hop: sender does not take a fee
    def cost_function_no_fees(self, G, amount, u, v):
        return amount * G.edges[v, u]["Delay"] * self.C_RISK_FACTOR + self.RISK_BIAS

    # construct route using C-lightning algorithm (uses ordinary dijkstra)
    def routePath(self, G, u, v, amt, payment_source=True, target_delay = 0 ):
        path, delay, amount, dist =  self.Dijkstra(G, u, v, amt, payment_source, target_delay)
        return {"path": path, "delay": delay, "amount": amount, "dist": dist}