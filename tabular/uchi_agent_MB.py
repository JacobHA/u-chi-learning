"""
Model-based algorithm, iterates over every initial state and action manually.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from frozen_lake_env import ModifiedFrozenLake, MAPS
from gymnasium.wrappers import TimeLimit
from tabular_utils import chi, get_dynamics_and_rewards, printf, solve_unconstrained, solve_unconstrained_v1, get_mdp_transition_matrix
from visualization import plot_dist, save_err_plot, save_policy_plot, save_thetas, save_u_plot

def logu_solver(env_src=None, map_name=None, beta=10):
    # Assuming deterministic dynamics only for now:
    # beta = 10
    max_steps = 1000

    if env_src is None:
        if map_name is not None:
            print("Defaulting to standard parameters.")
            n_action = 5
            desc = np.array(MAPS[map_name], dtype='c')
            env_src = ModifiedFrozenLake(
                n_action=n_action, max_reward=-0, min_reward=-1,
                step_penalization=1, desc=desc, never_done=False, cyclic_mode=False,
                # between 0. and 1., a probability of staying at goal state
                # an integer. 0: deterministic dynamics. 1: stochastic dynamics.
                slippery=0,
            )
    env = TimeLimit(env_src, max_episode_steps=max_steps)


    dynamics, rewards = get_dynamics_and_rewards(env)
    n_states, SA = dynamics.shape
    n_actions = int(SA / n_states)
    prior_policy = np.ones((n_states, n_actions)) / n_actions

    solution = solve_unconstrained(
        beta, dynamics, rewards, prior_policy, eig_max_it=1_000_000, tolerance=1e-12)
    l_true, u_true, v_true, optimal_policy, optimal_dynamics, estimated_distribution = solution
    rewards = (np.array(rewards)[0]).reshape(n_states, n_actions)
    theta_true = -np.log(l_true) / beta
    # get the twisted matrix:
    twisted_matrix = get_mdp_transition_matrix(dynamics, prior_policy)
    # twist w e^beta r:
    twisted_matrix = twisted_matrix.A @ np.diag(np.exp(beta * rewards.flatten()))
    # get ev gap:

    # Must "wisely" choose a reference state to normalize u by.
    u_ref_state = (0, 1)
    # where does the reference state send you to?
    chi_ref_state = int(dynamics[u_ref_state])

    delta_rwds = rewards - rewards[u_ref_state]

    next_state = dynamics.A[u_ref_state]


    init_u = np.ones((n_states, n_actions))
    init_chi = chi(init_u, n_states, n_actions, prior_policy=prior_policy)
    # init_u = u


    # Learning rate and training episodes:
    alpha = 0.02
    n_iteration = 600

    # Track eigenvalue during learning (can use to check convergence):
    thetas = []

    # We will be updating log(u) and chi on each iteration
    logu = np.log(init_u)
    ch = init_chi

    # Also can track error for convergence:
    errs = []

    for i in range(n_iteration):
        printf('Iteration', i, n_iteration)
        # Set old logu to current logu (so we can learn and calculate error between the two)
        loguold = logu
        # Loop over all state action pairs
        for state in range(n_states):
            for action in range(n_actions):
                if (state, action) != u_ref_state:
                    # Check which states will be transitioned to: (det. dynamics so only 1 possible next state)

                    state_prime = np.argwhere(
                        dynamics.A.T[state*env.nA + action] == 1)[0][0]

                    # Update log(u) based on the u-chi relationship
                    logu[state, action] = (
                        beta * delta_rwds[state, action] + np.log(ch[state_prime]/ch[chi_ref_state]))

        # Learn logu update
        logu = loguold * (1 - alpha) + alpha * logu
        logu -= logu[u_ref_state]

        # Update chi at each state-iteration (because it will change each time)
        ch = chi(np.exp(logu), n_states, n_actions, prior_policy=prior_policy)

        # Calculate error between old and new logu
        # chisa = np.array([np.log(ch)]*n_actions).T
        # errs.append(np.abs(logu - (delta_rwds + np.log(chisa))).sum())

        errs.append(np.abs(logu.flatten() - np.log(u_true)).sum())

        # Track eigenvalue
        theta = - beta * rewards[u_ref_state] - np.log(ch[chi_ref_state])
        thetas.append(theta)

    save_err_plot(errs, 'MB')
    save_u_plot(env, logu, u_true, prior_policy=prior_policy, name='MB')
    save_thetas(thetas, l_true, name='MB')


    pi_learned = np.exp(logu)/np.exp(logu).sum(axis=1, keepdims=True)

    u_true = u_true.A
    u_true = u_true.reshape(env.nS, env.nA)
    optimal_policy = u_true * prior_policy
    optimal_policy = optimal_policy / optimal_policy.sum(axis=1, keepdims=True)
    desc = env.unwrapped.desc.astype('U')
    save_policy_plot(desc, pi_learned, optimal_policy, name='MB')

    print(l_true)
    print(np.exp(-np.mean(thetas[-10:])))
    print('theta:', -np.log(l_true)/beta)
    print(u_true)

    # Run the learned policy:
    n_episode = 50
    r_list = []
    n_action = env.action_space.n
    for _ in range(n_episode):
        s, _ = env.reset()
        done = False
        r = 0
        while not done:
            # sample an action from pi[s]:
            a = np.random.choice(n_action, p=pi_learned[s][0])
            s, reward, terminated, truncated, _ = env.step(a)
            r += reward
            done = terminated or truncated
        r_list.append(r)

    r_avg = np.mean(r_list)
    r_std = np.std(r_list)
    print('Average reward:', r_avg)
    print('Standard deviation:', r_std)

    return r_avg, r_std

if __name__ == '__main__':
    logu_solver(map_name='7x7wall')