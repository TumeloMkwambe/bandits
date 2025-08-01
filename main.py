import numpy as np
from run import Run
import matplotlib.pyplot as plt
from algorithm import Greedy, Epsilon, UCB

num_actions = 10
num_timesteps = 1000
num_runs = 100

q_1 = 5
epsilon = 0.1
c = 2

Run_ = Run(num_actions, num_timesteps)

for run in range(num_runs):
    Run_.initial_values()
    Greedy_ = Greedy("greedy", num_actions, q_1)
    Epsilon_ = Epsilon("epsilon", num_actions, epsilon)
    UCB_ = UCB("ucb", num_actions, c)

    for algorithm in [Greedy_, Epsilon_, UCB_]:
        for t in range(num_timesteps):
            action = algorithm.select_action(t)
            reward = Run_.update_rewards(algorithm.name, action, t)
            algorithm.update(action, reward, t)

Run_.average_rewards(num_runs)

timesteps = np.arange(num_timesteps)

plt.figure(figsize=(10, 5))
plt.plot(timesteps, Run_.greedy_rewards, label='Greedy')
plt.plot(timesteps, Run_.epsilon_rewards, label='ϵ-Greedy')
plt.plot(timesteps, Run_.ucb_rewards, label='UCB')
plt.xlabel('Time Step')
plt.ylabel('Average Reward')
plt.title('Average Reward Over Time For Different Action Selection Methods')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()