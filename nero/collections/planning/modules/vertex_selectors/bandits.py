import pyro
import math
import random

class VertexSelectorPolicy(object):
    def __init__(self):
        pass

    def update(self, Y):
        pass

    def choose_arm(self, context=None):
        pass

class RandomVertexSelector(VertexSelectorPolicy):
    """
    Implementation of a Random Sampler where choosing each arm has equal probability
    """
    def __init__(self, num_actions: int):
        self.num_actions = num_actions
        self.params = [{"w": 0, "l": 0} for _ in range(self.num_actions)]
        self.arm_selected = 0
        
    def update(self, Y):
        if Y == 1:
            self.params[self.arm_selected]["w"] += 1
        else:
            self.params[self.arm_selected]["l"] += 1
            
    def choose_arm(self, context=None):
        # each arm has an equal chance of being selected
        probs = [1/self.n_bandits] * self.n_bandits
        self.arm_selected = sample("arm_selected", dist.Categorical(tensor(probs)))
        return self.arm_selected


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

	self.reward = reward
	self.num_actions = len(reference_graph_vertices)

        # exp3: int, (int, int -> float), float -> generator
        # perform the exp3 algorithm.
        # numActions is the number of actions, indexed from 0
        # rewards is a function (or callable) accepting as input the action and
        # producing as output the reward for that action
        # gamma is an egalitarianism factor
        weights = [1.0] * num_actions

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


