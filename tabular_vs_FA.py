import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium.wrappers import TimeLimit
import sys
sys.path.append("darer")
from LogUAgent import LogUAgent
from UAgent import UAgent
from darer.hparams import *
from darer.utils import get_eigvec_values
from tabular.tabular_utils import get_dynamics_and_rewards, solve_unconstrained
from tabular.frozen_lake_env import ModifiedFrozenLake, MAPS

config = maze
config.pop('beta')
map_name = '3x5uturn'
# map_name = 'hallway1'

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
    np.save(f'tabular/tabular_expt/data/{map_name}_{beta}u_true.npy', u_true)
    return -np.log(l_true) / beta

def FA_solution(beta, env):
    # Use an agent to solve the environment

    agent = UAgent(env, **config, log_interval=1000, tensorboard_log='pong',
                        num_nets=2, device='cpu',# use_rawlik=False,
                        beta=beta, render=False)#, aggregator='min')
    agent.learn(total_timesteps=250_000)
    get_eigvec_values(agent, save_name=f'tabular/tabular_expt/data/{map_name}_{beta}eigvec')
    # convert agent.theta to float
    theta = agent.theta.item()
    return theta

def main(beta):
    # initialize the environment
    n_action = 4
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

    exact = exact_solution(beta, env)#for beta in betas]
    print(exact)
    FA = FA_solution(beta, env) #for beta in betas

    # save the single beta data point with index (all scalar values):
    data = pd.DataFrame({'beta': beta, 'exact': exact, 'FA': FA}, index=[0])
    # get number of files with same prefix:
    num = len([f for f in os.listdir('tabular/tabular_expt/data') if f.startswith(f'{map_name}tabular_vs_FA')])
    data.to_csv(f'tabular/tabular_expt/data/{map_name}_{beta}_tabular_vs_FA_{num}.csv', index=False)

    plot_eigvec(beta)


def plot_eigvec(beta):
    # Plot the eigenvectors:
    true_eigvec = np.load(f'tabular/tabular_expt/data/{map_name}_{beta}u_true.npy')
    # fa_logeigvec = np.load(f'tabular/tabular_expt/data/{map_name}eigvec.npy')
    fa_eigvec = np.load(f'tabular/tabular_expt/data/{map_name}_{beta}eigvec.npy')
    fa_u = fa_eigvec.flatten()
    # fa_u = np.exp(fa_logeigvec.flatten())
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
    plt.savefig(f'tabular/tabular_expt/results/{map_name}_{beta}_eigvec.png')



def plot():
    # Plot the data in all CSV files
    # look for all files beginning with "map_name"tabular_vs_FA...:
    import glob
    files = glob.glob(f'tabular/tabular_expt/data/{map_name}tabular_vs_FA*.csv')
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
    plt.savefig(f'tabular/tabular_expt/results/{map_name}tabular_vs_FA.png')




if __name__ == '__main__':
    import multiprocessing as mp

    # pool = mp.Pool(15)
    #looks like levels off around beta=100, beta=0.1
    alphas = np.linspace(1/50, 10, 10)[::-1]
    betas = 1 / alphas
    betas = np.linspace(1,30,10)
    # pool.map(main, betas)
    # pool.close()
    # pool.join()
    for beta in betas:
        main(beta)
    
    plot()