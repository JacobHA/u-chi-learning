import os
import numpy as np
from frozen_lake_env import ModifiedFrozenLake, MAPS
from tabular_utils import softq_solver, get_dynamics_and_rewards, get_mdp_generator
from gymnasium.wrappers import TimeLimit
import matplotlib.pyplot as plt
from visualization import plot_dist
from uchi_agent_MB import logu_solver
from multiprocessing import Pool
import seaborn as sns
# sns.set_style('dark')
sns.set_context('paper', font_scale=2., rc={'lines.linewidth': 3.5})
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'serif'
# bold face:
# matplotlib.rcParams['font.weight'] = 'bold'

BETA = 15

def env_setup(map_name='hallway1', n_action=5):
    # initialize the environment
    desc = np.array(MAPS[map_name], dtype='c')
    env_src = ModifiedFrozenLake(
        n_action=n_action, max_reward=0, min_reward=-1,
        step_penalization=1, desc=desc, never_done=False, cyclic_mode=True,
        goal_attractor=0.,
        # between 0. and 1., a probability of staying at goal state
        # an integer. 0: deterministic dynamics. 1: stochastic dynamics.
        slippery=0,
    )
    return env_src

def calculate_timescale(env_src, beta=BETA):
    dynamics_matrix, rewards = get_dynamics_and_rewards(env_src)
    # find the second largest eigenvalue of the dynamics matrix:
    prior_policy = np.ones((env_src.nS, env_src.nA)) / env_src.nA
    mdp_generator = get_mdp_generator(env_src, dynamics_matrix, prior_policy)
    eigenvalues, _ = np.linalg.eig(mdp_generator.A)
    eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
    gap = 1 - eigenvalues[1]
    # find the natural timescale:
    std_rl_timescale = 1 / gap

    tilted_matrix = mdp_generator.A @ np.diag(np.exp(beta * rewards.A[0]))
    eigenvalues, eigenvecs = np.linalg.eig(tilted_matrix)
    # sort both by magnitude:
    eigenvalues, eigenvecs = zip(*sorted(zip(eigenvalues, eigenvecs), key=lambda x: np.abs(x[0]), reverse=True))
    eig0, eig1 = eigenvalues[0:2]
    # eig0, eig1 = np.sort(np.abs(eigenvalues))[::-1][0:2]
    # dominant left eigenvector:
    u = eigenvecs[0]
    gap = eig0 / eig1# find the natural timescale:
    beta_timescale = 1/np.log(gap)

    # driven_matrix = np.diag(u**(-1)) @ tilted_matrix @ np.diag(u) / eig0
    # # get driven matrix eigenvalues:
    # eigenvalues, _ = np.linalg.eig(driven_matrix)
    # eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
    # gap = 1 - eigenvalues[1]
    # # find the natural timescale:
    # driven_timescale = 1 / np.loggap


    print('timescale:', beta_timescale)
    return beta_timescale

def run(env_src, gamma, plot_gamma, beta=BETA, max_steps=2000,):
    env = TimeLimit(env_src, max_episode_steps=max_steps)
    n_action = env.action_space.n
    # solve the ent-reg objective with discounting:
    tol = np.finfo(float).eps
    Qi, Vi, pi, qs, errors_list = softq_solver(env, gamma=gamma, beta=beta, verbose=True, tolerance=tol)
    solve_time = len(errors_list)
    # calculate the optimal policy:
    q = Qi - Qi.max(axis=-1, keepdims=True)
    pi = np.exp(beta*q)
    pi /= pi.sum(axis=-1, keepdims=True)
    # run the optimal policy and compute the average reward:
    n_episode = 1000
    r_list = []
    # gather statistics on the states visited:
    s_visits = np.zeros(env.observation_space.n)
    for _ in range(n_episode):
        s, _ = env.reset()
        s_init = s
        done = False
        r = 0
        while not done:
            # sample an action from pi[s]:
            a = np.random.choice(n_action, p=pi[s][0])
            s, reward, terminated, truncated, _ = env.step(a)
            s_visits[s] += 1
            r += reward
            done = terminated or truncated
        r_list.append(r)
    r_avg = np.mean(r_list)
    # remove the starting state:
    s_visits[s_init] -= n_episode
    print(f'gamma={gamma:.3f}, r_avg={r_avg:.3f}, r_std={np.std(r_list):.3f}, solve_time={solve_time}')
    print(s_visits)
    if plot_gamma:
        np.save(f'{map_name}/{gamma}-state_dists.npy', s_visits)
    r_std = np.std(r_list)
    return r_avg, r_std, solve_time

def plot(horizons, returns, timescale, gs, speeds=None, return_stds=None, logu_rwd=None, logu_std=None):
    gammas = 1 - 1 / horizons
    returns = -np.array(returns)
    logu_rwd = -logu_rwd
    return_stds = np.array(return_stds)
    plt.figure(figsize=(10,6))
    # plt.xscale('log')
    plt.plot(gammas, returns, 'k', label='SQL')
    plt.fill_between(gammas, returns - return_stds, returns + return_stds, color='k', alpha=0.35)
    # do a linear smoothing:
    # returns = np.array(returns)
    # window = 3
    # smoothed_returns = np.convolve(returns, np.ones(window)/window, mode='valid')
    # smoothed_stds = np.convolve(return_stds, np.ones(window)/window, mode='valid')
    # smoothed_horizons = horizons[1:-1]
    # Do a fill_between:
    # return_stds = np.array(return_stds)
    # ax1.plot(smoothed_horizons, smoothed_returns, 'k', lw=1.5, label='Exact Soft-Q')

    # ax1.fill_between(smoothed_horizons, smoothed_returns - smoothed_stds, smoothed_returns + smoothed_stds, color='k', alpha=0.35)


    # plot a thick shaded band of width logu_std:
    min_gamma = gammas.min()
    max_gamma = gammas.max()
    all_x = np.linspace(min_gamma, max_gamma, 10)
    plt.fill_between(all_x, logu_rwd - logu_std, logu_rwd + logu_std, color='blue', alpha=0.3)  
    plt.axhline(logu_rwd, xmin=0.01, xmax=0.99, linestyle='-.', color='b', label='EVAL (our method)')

    # plot a vertical line at timescale
    mixing_gamma = 1 - 1 / timescale
    plt.axvline(mixing_gamma, linestyle='--', color='r', label='Spectral Gap', lw=2)
    # Add a text label:
    # plt.text(mixing_gamma*1.002, 38, r'Spectral Gap', color='r')#, fontsize=20)
    # plt.xlabel(r'Discounting Timescale, $\left(1-\gamma\right)^{-1}$')
    plt.xlabel(r'Discount Factor, $\gamma$')
    # plt.ylabel('Reward Accumulated \nby Optimal Policy')

    plt.ylabel('Average Steps to Goal')
    # make the y-axis label, ticks and tick labels match the line color.
    plt.title('Discounted vs. Average-Reward Objective\n', fontsize=24)
    plt.xlim(min_gamma, max_gamma)
    # Use horizon scaling:
    horizon_scaling=False
    if horizon_scaling:
        plt.xscale('function', functions=(forward, inverse))
        # place x ticks uniformly in transformed space:
        xmin, xmax = plt.xlim()
        horizons = forward(np.linspace(xmin, xmax, 5))
        # get uniform ticks on horizons based on xlim min and max:
        ticks = np.linspace(horizons.min(), horizons.max(), 5)
        # transform back to gammas:
        plt.xticks(inverse(ticks), [f'{1-1/t:.2f}' for t in ticks])

    # plt.ylim(6,40)
    # # make a legend manually:
    # legend_elements = [
    #     matplotlib.lines.Line2D([0], [0], color='k', lw=1.5, label='Exact Soft-Q'),
    #     matplotlib.lines.Line2D([0], [0], color='b', lw=1.5, label='Exact LogU'),
    #     matplotlib.lines.Line2D([0], [0], color='r', lw=2, linestyle='--', label='Spectral Gap'),
    # ]
    # plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.15, -0.15), fancybox=True, shadow=True, ncol=3)
    
    # Put legend on right side of fig, 1col:
    # plt.legend(loc='upper center', fancybox=True, shadow=True, ncol=3)
    # plt.legend(loc='upper right', fancybox=True, shadow=True, ncol=1)
    # Legend below figure:
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=3)
    plt.tight_layout()
    g1,g2=gs

    # add arrows:
    gammas = np.real_if_close(gammas)
    # point an arrow from bottom of fig to the nearest rewards of g1:
    end_points = [(g1, np.interp(g1, gammas, returns)), (g2, np.interp(g2, gammas, returns))]
    init_points = [(0.85, 105), (0.975, 105)]
    for init_point, end_point in zip(init_points, end_points):
        plt.annotate("", xy=end_point, xytext=init_point,
                     arrowprops=dict(arrowstyle="->", color='k', lw=2))


    

    # Add inset plots of map_name/gamma_{g1,g2}.png:
    # put g1 on the left of spectral gap:
    size=0.32
    axins1 = plt.axes([0.17, 0.5, size, size])
    axins1.imshow(plt.imread(f'{map_name}/gamma_{g1}.png'))
    axins1.axis('off')

    axins2 = plt.axes([0.685, 0.5, size, size])
    axins2.imshow(plt.imread(f'{map_name}/gamma_{g2}.png'))
    axins2.axis('off')
    

    plt.tight_layout()
    plt.savefig(f'{map_name}/time_v_reward.png', dpi=400)
    plt.close()

# Function for gamma transform
def forward(a):
    return 1 / (1 - a)

def inverse(a):
    return 1 - 1/a


def main(map_name='hallway1', gs=[0.87, 0.96]):
    env_src = env_setup(map_name=map_name)
    timescale = calculate_timescale(env_src, beta=BETA)

    returns = []
    return_stds = []
    speeds = []
    map_state_dists = []
    gamma_values = np.linspace(0.8, 0.999, 50)

    # horizons = np.linspace(1,55,5)
    # gamma_values = 1 - 1 / horizons

    # use a thread pool to run the experiments in parallel:
    os.makedirs(f'{map_name}', exist_ok=True)
    # plot gamma if it is min or max in range:
    plot_gammas = (gamma_values == gamma_values.min()) | (gamma_values == gamma_values.max())
    with Pool() as pool:
        results = pool.starmap(run, [(env_src, gamma, plot_gamma) for gamma, plot_gamma in zip(gamma_values, plot_gammas)])
    for r_avg, r_std, solve_time in results:
        returns.append(r_avg)
        return_stds.append(r_std)
        speeds.append(1 / solve_time)
    
    pool.close()

    horizons = 1 / (1 - gamma_values)
    # calculate the logu solution:
    logu_rwd, logu_std = logu_solver(env_src, beta=BETA)
    # save the data:
    timescale_tiled = np.tile(timescale, len(horizons))
    data = np.array([horizons, returns, speeds, return_stds, timescale_tiled])
    np.save(f'{map_name}/data.npy', data)
    plot(horizons, returns, timescale, gs=gs, speeds=speeds, return_stds=return_stds, logu_rwd=logu_rwd, logu_std=logu_std)

    
def plot_policies(map_name, gammas, max_steps=1000, beta=BETA):
    desc = np.array(MAPS[map_name], dtype='c')
    env_src = env_setup(map_name=map_name)
    env = TimeLimit(env_src, max_episode_steps=max_steps)
    if gammas is None:
        # get all of the gammas in map_name:
        gammas = np.array([float(file.split('-')[0]) for file in os.listdir(map_name) if file.endswith('state_dists.npy')])
    tol = np.finfo(float).eps
    for gamma in gammas:

        # assert gamma**max_steps < tol
        Qi, Vi, pi, qs, errors_list = softq_solver(env, gamma=gamma, beta=beta, verbose=True, tolerance=tol)
        # calculate the optimal policy:
        q = Qi - Qi.max(axis=-1, keepdims=True)
        pi = np.exp(beta*q)
        pi /= pi.sum(axis=-1, keepdims=True)
        # plot the policy:
        # plot_dist(desc, pi, main_title=f'gamma={gamma}', filename=f'gamma_{gamma}.png')
        # Load the state_dists:
        s_dist = np.load(f'{map_name}/{gamma}-state_dists.npy')
        # translate horizons to gammas:
        # find the index of the gamma we want:
        print(s_dist)
        plot_dist(desc, s_dist, filename=f'{map_name}/gamma_{gamma}.png', dpi=400, symbol_size=500)

def plot_from_data(map_name, gs):
    data = np.load(f'{map_name}/data.npy')
    horizons, returns, speeds, return_stds, timescale_tiled = data
    timescale = timescale_tiled[0]
    env_src = env_setup(map_name=map_name)
    logu_rwd, logu_std = logu_solver(env_src, beta=BETA)
    plot(horizons, returns, timescale, speeds=speeds, return_stds=return_stds, 
         logu_rwd=logu_rwd, logu_std=logu_std, gs=gs)

if __name__ == '__main__':
    # map_name = 'hallway2'
    map_name='9x9channel'
    gs = [0.87, 0.93]
    # main(map_name=map_name, gs=gs)
    # for g in gs:
        # run(env_setup(map_name=map_name), gamma=g, plot_gamma=True)

    # plot_policies(map_name=map_name, gammas=gs)
    plot_from_data(map_name, gs=gs)