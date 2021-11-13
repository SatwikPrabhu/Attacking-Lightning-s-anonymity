from queue import  PriorityQueue
from routingalgos.base import Routing
import nested_dict as nd
from math import inf
import random as rn

class ShadowRouting(Routing):

    # Initialize routing algorithm
    def __init__(self, base_routing) -> None:
        super().__init__()
        self.__baseRouting = base_routing

    # human readable name for this routing algorithm
    def name(self):
        return self.__baseRouting.name() + "with shadow routing"

    # Returns tech used by base routing  
    def tech(self):
        return self.__baseRouting.tech()

    # cost function, uses same cost function as base routing algorithm
    def cost_function(self, G, amount, u, v):
        return self.__baseRouting.cost_function(G, amount, u, v)


    # cost function for first hop, uses same cost function as base routing algorithm
    def cost_function_no_fees(self, G, amount, u, v):
        return self.__baseRouting.cost_function_no_fees(G, amount, u, v)

    def routePath(self, G, u, v, amt, payment_source=True, target_delay = 0 ):
        return self.__baseRouting.routePath(G, u, v, amt, payment_source, target_delay)

    def adversarial_attack(self, G,adversary,delay,amount,pre,next, attack_position = -1):
        return self.__baseRouting.adversarial_attack(G, adversary, delay, amount, pre, next, attack_position, shadow_routing = True)