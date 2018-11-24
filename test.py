from crazyenv.env import HoverEnv

env = HoverEnv()
obs = env.reset()

for trial in range(100): # 100 trials
    for _ in range(100): # 10 seconds:
        obs, reward, done, _ = env.step([0., 0., 1-obs[2]])
        env.render()
    print('finished trial #{}'.format(trial))
    obs = env.reset()

env.close()
