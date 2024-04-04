# Plot the u function over time in training


# Do finetuning on the Acrobot UAgent
import gymnasium
import numpy as np
import wandb
import argparse
import yaml
import sys

from utils import safe_open

sys.path.append('darer')
from SoftQAgent import SoftQAgent
from CustomDQN import CustomDQN
from UAgent import UAgent

env_id = 'Acrobot-v1'
algo = 'u'

env_to_steps = {
    'Acrobot-v1': 3000,
    'CartPole-v1': 2_000,
}

full_config = {}
default_params = safe_open(f'hparams/{env_id}/{algo}.yaml')
full_config.update(default_params)
AGG = 'max'
full_config['aggregator'] = AGG
NUM_NETS = 2
steps = 100

agent = UAgent(env_id, **full_config,
                device='auto', log_interval=env_to_steps[env_id] // steps,
                tensorboard_log=f'ft_logs/{env_id}', 
                num_nets=NUM_NETS,
                render=False,
                )

# get the u function and plot it
bounds = agent.env.observation_space.low, agent.env.observation_space.high
states = np.linspace(bounds[0], bounds[1], 100)
n_actions = agent.env.action_space.n
import matplotlib.pyplot as plt
state_dim = 1#states.shape[1]

for step in range(steps):
    print("Plotting after step", step)
    u = [net(states) for net in agent.model.nets]
    # Plot the first state index for now for each action:
    # use multiple axes for the different state dimensions:
    fig, axes = plt.subplots(state_dim, 1, figsize=(10, 10*state_dim/3))
    if state_dim == 1:
        axes = [axes]
    for state_idx, ax in enumerate(axes):
        for action, color in zip(range(n_actions), ['r','b']):
            for unet in u:
                ax.plot(states[:,state_idx], unet[:,action].detach().cpu(), 
                        label=f'Action {action}', color=color)
        # for stability in gif:
        ax.set_ylim(0,3)
    plt.suptitle(f'U function at step {step}. Reward: {agent.avg_eval_rwd}')
    plt.legend()
    plt.savefig(f'experiments/u_func_plots/{AGG}_{NUM_NETS}_{step}.png')
    plt.close()
    # continue training:
    agent.env_steps = 0
    agent.learn(total_timesteps=env_to_steps[env_id] // steps)



# Now turn into a gif:
import imageio
images = []
for i in range(steps):
    images.append(imageio.v2.imread(f'experiments/u_func_plots/{AGG}_{NUM_NETS}_{i}.png'))
imageio.mimsave(f'experiments/u_func_plots/{env_id}_{AGG}_{NUM_NETS}.gif', images)

# now delete all the images:
import os
for i in range(steps):
    os.remove(f'experiments/u_func_plots/{AGG}_{NUM_NETS}_{i}.png')