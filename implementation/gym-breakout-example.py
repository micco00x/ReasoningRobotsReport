import gym

env = gym.make("BreakoutNoFrameskip-v4")
env.reset()

for _ in range(1000):
  env.render()
  action = env.action_space.sample() # takes random actions
  observation, reward, done, info = env.step(action)
  if done == True:
    env.reset()

env.close()
