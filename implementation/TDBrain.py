from abc import abstractmethod

import numpy as np
from gym.spaces import Discrete

from rltg.agents.brains.Brain import Brain
from rltg.agents.parameters.Eligibility import Eligibility
from rltg.agents.parameters.Parameter import Constant, AlphaVisitDecay
from rltg.agents.policies.EGreedy import EGreedy
from rltg.agents.policies.Policy import Policy
from rltg.utils.misc import mydefaultdict, AgentObservation


class TDBrain(Brain):
    def __init__(self, observation_space:Discrete, action_space:Discrete, policy:Policy=EGreedy(),
                 gamma=0.99, alpha=None, lambda_=0):
        super().__init__(observation_space, action_space, policy)

        self.gamma = gamma
        self.alpha = Constant(alpha) if alpha is not None else AlphaVisitDecay(action_space)
        self.lambda_ = lambda_

        self._init()

    def _init(self):
        # sparse representation
        self.Q = mydefaultdict(np.zeros((self.action_space.n,)))
        self.eligibility = Eligibility(self.lambda_, self.gamma)

    def start(self, state):
        super().start(state)
        self.eligibility.reset()
        action = self.choose_action(state)
        return action

    def choose_action(self, state, optimal=False):
        Q_values = self.Q[state] # defaultdict, look at __init__
        action = self.policy.choose_action(Q_values, optimal=optimal or self.eval)
        return action

    def step(self, obs:AgentObservation, *args, **kwargs):
        if self.eval:
            return self.choose_action(obs.state2, optimal=self.eval)

        action2 = self.update_Q(obs)
        return action2

    def end(self, obs:AgentObservation, *args, **kwargs):
        super().end(obs)
        if self.eval:
            return
        state, action, reward, state2 = obs.unpack()
        delta = reward - self.Q[state][action]
        for (s, a) in self.eligibility.traces:
            self.Q[s][a] += self.alpha.get(s,a) * delta * self.eligibility.get(s, a)

    @abstractmethod
    def update_Q(self, obs: AgentObservation):
        raise NotImplementedError

    def observe(self, obs:AgentObservation, *args, **kwargs):
        super().observe(obs)
        self.eligibility.to_one(obs.state, obs.action)
        self.alpha.update(obs.state, obs.action)

    def reset(self):
        self._init()
