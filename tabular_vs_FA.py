import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tabular.frozen_lake_env import ModifiedFrozenLake, MAPS
from gymnasium.wrappers import TimeLimit

from tabular.tabular_utils import get_dynamics_and_rewards, solve_unconstrained


from darer.MultiLogU import LogULearner
from darer.hparams import *
from utils import get_eigvec_values
config = cartpole_hparams2
config.pop('beta')
map_name = '4x4'
def exact_solution(beta, env):

    dynamics, rewards = get_dynamics_and_rewards(env.unwrapped)
    n_states, SA = dynamics.shape
    n_actions = int(SA / n_states)
    prior_policy = np.ones((n_states, n_actions)) / n_actions

    solution = solve_unconstrained(
        beta, dynamics, rewards, prior_policy, eig_max_it=1_000_000, tolerance=1e-12)
    l_true, u_true, v_true, optimal_policy, optimal_dynamics, estimated_distribution = solution

    print(f"l_true: {l_true}")
    # save u true:
    np.save(f'{map_name}u_true.npy', u_true)
    return -np.log(l_true) / beta

def FA_solution(beta, env):
    # Use MultiLogU to solve the environment

    agent = LogULearner(env, **config, log_interval=1000, log_dir='pend',
                        num_nets=2, device='cpu', 
                        beta=beta, render=0, aggregator='max')
    agent.learn(total_timesteps=100_000)
    get_eigvec_values(agent, save_name=f'{map_name}eigvec')
    # convert agent.theta to float
    theta = agent.theta.item()
    return theta

def main():
    # initialize the environment
    n_action = 5
    max_steps = 200
    desc = np.array(MAPS[map_name], dtype='c')
    env_src = ModifiedFrozenLake(
        n_action=n_action, max_reward=0, min_reward=-1,
        step_penalization=1, desc=desc, never_done=True, cyclic_mode=True,
        # between 0. and 1., a probability of staying at goal state
        # an integer. 0: deterministic dynamics. 1: stochastic dynamics.
        slippery=0,
    )
    env = TimeLimit(env_src, max_episode_steps=max_steps)

    # Set the beta values to test
    betas = np.logspace(1, 1, 4)
    betas = [14,19,24,30,37]
    betas = np.linspace(1, 50, 50)
    betas = [10]

    exact = [exact_solution(beta, env) for beta in betas]
    print(exact)
    FA = [FA_solution(beta, env) for beta in betas]

    # save the data:
    data = pd.DataFrame({'beta': betas, 'exact': exact, 'FA': FA})
    data.to_csv(f'{map_name}tabular_vs_FA50.csv', index=False)
# [0.9999155228491464, 0.9987485370048285, 0.9891983268590409, 0.9777515697268231, 0.9681814068991201, 0.9603645308274785
    plt.figure()
    plt.plot(betas, exact, 'ko-', label='Exact')
    plt.plot(betas, FA, 'bo', label='FA')
    plt.legend()
    plt.savefig(f'{map_name}tabular_vs_FA.png')

def plot():
    # Plot the data in all CSV files
    # look for all files beginning with "map_name"tabular_vs_FA...:
    import glob
    files = glob.glob(f'{map_name}tabular_vs_FA*.csv')
    # concat the csvs in a single df:
    data = pd.concat([pd.read_csv(f) for f in files])
    # Sort by beta
    data = data.sort_values(by=['beta'])
    # tske mean of the data with the same beta
    data = data.groupby(['beta']).mean().reset_index()
    betas = data['beta'].values
    exact = data['exact'].values
    FA = data['FA'].values
    plt.figure()
    plt.plot(betas, exact, 'ko-', label='Exact')
    plt.plot(betas, FA, 'ro', label='FA')
    plt.xlabel
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$\theta$')
    plt.title(f'Free energy vs. inverse temperature on {map_name}')
    plt.legend()
    plt.savefig(f'{map_name}tabular_vs_FA.png')

    # Plot the eigenvectors:
    true_eigvec = np.load(f'{map_name}u_true.npy')
    fa_logeigvec = np.load(f'{map_name}eigvec.npy')
    fa_u = np.exp(fa_logeigvec.flatten())
    true_u = true_eigvec.flatten()
    # normalize them to have the same norm:
    true_u /= np.linalg.norm(true_u)
    fa_u /= np.linalg.norm(fa_u)
    plt.figure()
    plt.plot(true_u, 'ko-', label='Exact')
    plt.plot(fa_u, 'ro', label='FA')
    plt.xlabel('State-action pairs')
    plt.ylabel('Eigenvector')
    plt.title(f'Eigenvectors on {map_name}')
    plt.legend()
    plt.savefig(f'{map_name}eigvec.png')


if __name__ == '__main__':
    main()
    plot()