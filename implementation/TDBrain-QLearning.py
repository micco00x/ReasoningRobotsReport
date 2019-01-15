class QLearning(TDBrain):
    def __init__(self, observation_space:Discrete, action_space, policy:Policy=EGreedy(),
                 gamma=0.99, alpha=None, lambda_=0):
        super().__init__(observation_space, action_space, policy, gamma, alpha, lambda_)

    def update_Q(self, obs:AgentObservation):
        state, action, reward, state2 = obs.unpack()

        action2 = self.choose_action(state2)
        Qa = np.max(self.Q[state2])
        actions_star = np.argwhere(self.Q[state2] == Qa).flatten().tolist()

        delta = reward + self.gamma * Qa - self.Q[state][action]
        for (s, a) in set(self.eligibility.traces.keys()):
            self.Q[s][a] += self.alpha.get(s,a) * delta * self.eligibility.get(s, a)
            if action2 in actions_star:
                self.eligibility.update(s, a)
            else:
                self.eligibility.to_zero(s, a)

        return action2
