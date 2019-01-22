env = gym.make("BreakoutNoFrameskip-v4")
env = EpisodicLifeEnv(env)
env = FireResetEnv(env)
env = MaxAndSkipEnv(env, skip=4)
