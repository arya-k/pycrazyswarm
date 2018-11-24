from crazyenv.env import HoverEnv

env = HoverEnv()
obs = env.reset()

for _ in range(36000):
    obs, reward, done, _ = env.step([0., 0., 1-obs[2]])
    env.render()
    print(reward)

env.close()
