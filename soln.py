from crazyenv.env import HoverEnv

env = HoverEnv()

obs = env.reset()
total_reward = 0.0

for _ in range(512):
    obs, reward, done, _ = env.step([0., 0., 1-obs[2]])
    total_reward += reward

print(total_reward)
