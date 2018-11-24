from crazyenv.env import HoverEnv

env = HoverEnv()

for trial in range(1000): # 1000 trials
    obs = env.reset()
    for _ in range(100): # 10 seconds:
        obs, reward, done, _ = env.step([0., 0., 1-obs[2]])
        #env.render()
#env.close()
