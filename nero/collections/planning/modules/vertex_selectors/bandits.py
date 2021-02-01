
from typing import List, Tuple, Callable, Optional

import torch
import pyro
import pyro.distributions as dist
from pyro import sample
import math
import random


class VertexSelectorPolicy(object):
    def __init__(self, num_vertices: int) -> None:
        self.num_bandits = num_vertices
        self.initialized = torch.zeros(self.num_bandits)
        self.rewards = torch.zeros(self.num_bandits) 
        self.p_bandits = dist.Categorical(torch.tensor([1/self.num_bandits]*self.num_bandits)) # Discrete Uniform
        self.prev_selected_bandit_idx = -1 

    def update_vertex(self, reward: float, vertex: Optional[int] = None) -> None:
        pass

    def select_vertex(self, active_bandits: List[int]) -> int:
        pass

    def reward_float_to_bool(self, reward: float, threshold: float) -> bool:
        if reward > threshold:
            return True
        else:
            return False

    def reset(self) -> None:
        self.rewards = torch.zeros(self.num_bandits) 
        #self.p_bandits = dist.Categorical(torch.tensor([1/self.num_bandits]*self.num_bandits)) # Discrete Uniform
        self.prev_selected_bandit_idx = -1 


class RandomVertexSelector(VertexSelectorPolicy):
    """
    Implementation of a Random Selector where choosing each bandit arm has equal probability.
    """
    def __init__(self, num_vertices: int) -> None:
        super().__init__(num_vertices)
        
    def update_vertex(self, reward: float, vertex_idx: Optional[int] = None) -> None:
        pass
            
    def select_vertex(self, active_bandits: List[int]) -> int:
        selected_bandit_idx = sample("selected_bandit_idx", self.p_bandits).item()
        while selected_bandit_idx not in active_bandits:
            selected_bandit_idx = sample("selected_bandit_idx", self.p_bandits).item()
        self.prev_selected_bandit_idx = selected_bandit_idx
        return selected_bandit_idx

class BetaVertexSelector(VertexSelectorPolicy):
    """
    Model each bandit as a beta distribution and run a form of Thompson sampling. 
    """
    def __init__(self, num_vertices: int, r_thresh: float = 10.) -> None:
        super().__init__(num_vertices)
        self.r_thresh = r_thresh 
        self.p_bandits = torch.zeros(self.num_bandits)
        self.params = torch.ones((self.num_bandits,2))
        
        
    def update_vertex(self, reward: float, vertex_idx: int = None, increment: int = 1) -> None:
        pass

            
    def select_vertex(self, active_bandits: List[int]) -> int:
        # check that active bandits is not empty
        assert active_bandits

        for i in range(self.num_bandits):
            self.p_bandits[i] = pyro.sample(f"bandit{i}_cost", dist.Beta(self.params[i,0], self.params[i,1]))
        _, ranked_bandit_idxs = torch.topk(self.p_bandits, self.num_bandits)
        ranked_bandit_idxs = ranked_bandit_idxs.cpu().numpy().tolist()

        best_valid_bandit_idx = None
        print(f"RANKED BANDITS: {ranked_bandit_idxs}")
        print(f"ACTIVE BANDITS: {active_bandits}")
        for idx in ranked_bandit_idxs:
            if idx in active_bandits:
                best_valid_bandit_idx = idx
                break

        # update bandits
        # only increment beta when the bandit is pulled
        self.params[best_valid_bandit_idx,1] += 1

        assert best_valid_bandit_idx
        return best_valid_bandit_idx

class EpsilonGreedyVertexSelector(VertexSelectorPolicy):
    """
    Greedily selects the highest reward bandit (1-epsilon)% of the time.
    Explores epsilon% of the time by taking a random action (uniform).
    """
    def __init__(self, num_vertices: int, epsilon: float) -> None:
        super().__init__(num_vertices)
        self.p_explore = dist.Bernoulli(epsilon) # prob of exploring 
        
    def update_vertex(self, reward: float, vertex_idx: Optional[int] = None) -> None:
        if vertex_idx:
            self.rewards[vertex_idx] = reward
        else:
            self.rewards[self.prev_selected_bandit_idx] = reward
            
    def select_vertex(self) -> int:
        explore = bool(sample("explore", self.p_explore).item())

        if explore:
            selected_bandit_idx = sample("selected_bandit_idx", self.p_bandits).item()
        else:
            selected_bandit_idx = torch.argmax(self.rewards).item()
            # could return the top k instead
            #selected_bandit_idx = torch.topk(self.rewards, k)

        self.prev_selected_bandit_idx = selected_bandit_idx
        return selected_bandit_idx

#class BernoulliBanditVertexSelector() 


class UCB1VertexSelector(VertexSelectorPolicy):
    """
    Implements UCB1 where score is maintained internally based on reward updates.
    """
    def __init__(self, num_vertices: int, epsilon: float) -> None:
        super().__init__(num_vertices)    
        self.timestep = 0 # introduce time as state in UCB1 (num rounds so far)
        
    def update_vertex(self, reward: float, vertex_idx: Optional[int] = None) -> None:
        if vertex_idx:
            self.rewards[vertex_idx] = reward
        else:
            self.rewards[self.prev_selected_bandit_idx] = reward
            
    def select_vertex(self) -> int:
        self.timestep += 1

        explore = bool(sample("explore", self.p_explore).item())

        if explore:
            selected_bandit_idx = sample("selected_bandit_idx", self.p_bandits).item()
        else:
            selected_bandit_idx = torch.argmax(self.rewards).item()
            # could return the top k instead
            #selected_bandit_idx = torch.topk(self.rewards, k)

        self.prev_selected_bandit_idx = selected_bandit_idx
        return selected_bandit_idx


class ThompsonVertexSelector(VertexSelectorPolicy):
    """
    Implementation of Thompson Sampling where our prior is adjusted based on reward assigned to a vertex that.
    """
    def __init__(self, num_vertices: int) -> None:
        self.num_bandits = num_vertices
        self.rewards = torch.zeros((self.num_bandits))
        self.p_bandits = dist.Categorical(torch.tensor([1/self.num_bandits]*self.num_bandits))
        self.prev_selected_bandit_idx = None
        
    def update_vertex(self, reward: float, vertex_idx: Optional[int] = None) -> None:
        pass
            
    def select_vertex(self) -> int:
        pass


class Exp3VertexSelector:
    """
    The Exp3 algorithm (https://cseweb.ucsd.edu/~yfreund/papers/bandits.pdf p.6)
    applied as a vertex selector given a reference graph.

    Modified with inspiration from:
    https://github.com/j2kun/exp3/blob/main/exp3.py
    https://github.com/ucla-csl/mabuc/blob/master/src/main/matlab/exp3Run.m
    """
    def __init__(
        self,
        reference_graph_vertices: List[Tuple[float]],
	reward: Callable = None,
	gamma: float = 0.5,
	reward_min: float = 0.0,
	reward_max: float = 0.1,
        ) -> None:
        # reward is cost reduction

        self.reward = reward
        self.num_actions = len(reference_graph_vertices)

        # exp3: int, (int, int -> float), float -> generator
        # perform the exp3 algorithm.
        # numActions is the number of actions, indexed from 0
        # rewards is a function (or callable) accepting as input the action and
        # producing as output the reward for that action
        # gamma is an egalitarianism factor
        weights = [1.0] * self.num_actions

        t = 0
        while True:
            p = self.distr(weights, gamma)
            choice = self.draw(probabilityDistribution)
            theReward = self.reward(choice, t)
            scaledReward = (theReward - rewardMin) / (rewardMax - rewardMin) # rewards scaled to 0,1

            estimatedReward = 1.0 * scaledReward / probabilityDistribution[choice]
            weights[choice] *= math.exp(estimatedReward * gamma / self.num_actions) # important that we use estimated reward here!

            yield choice, theReward, estimatedReward, weights
            t = t + 1

    def draw(weights):
        choice = random.uniform(0, sum(weights))
        choiceIndex = 0

        for weight in weights:
            choice -= weight
            if choice <= 0:
                return choiceIndex

            choiceIndex += 1

    # distr: [float] -> (float)
    # Normalize a list of floats to a probability distribution.  Gamma is an
    # egalitarianism factor, which tempers the distribtuion toward being uniform as
    # it grows from zero to one.
    def distr(weights, gamma=0.0):
        theSum = float(sum(weights))
        return tuple((1.0 - gamma) * (w / theSum) + (gamma / len(weights)) for w in weights)

    def mean(aList):
       theSum = 0
       count = 0

       for x in aList:
          theSum += x
          count += 1

       return 0 if count == 0 else theSum / count
         

if __name__ == '__main__':
    vs = BetaVertexSelector(num_vertices=10)
    for i in range(10):
        print(vs.select_vertex([i for i in range(0,10)]))
    for i in range(100):
        vs.update_vertex(200, 5)
    for i in range(10):
        print(vs.select_vertex([i for i in range(0,10)]))
    import ipdb; ipdb.set_trace()

