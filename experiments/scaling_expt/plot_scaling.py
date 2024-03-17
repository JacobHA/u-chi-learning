# Plot the results from scaling_expt.py
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
sns.set_theme(style="darkgrid")

folder = 'experiments/scaling_expt/results'
# env_id = 'Acrobot-v1'
env_id = 'CartPole-v1'

# Plot:
plt.figure()
for algo in ['u', 'dqn', 'ppo']:
# algo = 'u'
    try:
        df = pd.read_csv(os.path.join(folder, f'{env_id}-{algo}.csv'), header=None)
    except FileNotFoundError:
        print(f"File not found for {algo}")
        continue
    # Average the last 5 columns for a given row:
    # Name the first column hidden_dim:
    num_cols = df.shape[1]
    df.columns = ['hidden_dim'] + list(range(num_cols-1))
    df = df.sort_values(by='hidden_dim')
    # take average over same hidden_dim:
    df = df.groupby('hidden_dim')
    # Get mean and std:
    df = df.agg(['mean', 'std'])
    df = df.reset_index()
    # df.columns = ['hidden_dim', 'auc', 'auc_std']
    # Average over the means in each row:
    df['avg_final'] = df[[(i, 'mean') for i in range(51)]].mean(axis=1)
    df['avg_final_std'] = df[[(i, 'std') for i in range(51)]].mean(axis=1)

    # mean:
    # plt.plot(df['hidden_dim'], df['auc'], 'ko', label='mean')
    # std as error bars (center the bars on the mean):
    # replace nan stds with 0:
    # df['auc_std'] = df['auc_std'].fillna(0)
    # plt.errorbar(df['hidden_dim'], df['avg_final'], df['avg_final_std'], label=f'{algo}', fmt='o')
    # Plot shaded region for std:
    plt.plot(df['hidden_dim'], df['avg_final'], 'o-', label=f'{algo}')
    plt.fill_between(df['hidden_dim'], df['avg_final']-df['avg_final_std'], df['avg_final']+df['avg_final_std'], alpha=0.3)

    # Draw a star at 64 (where hparam was optimized), plot on top of line:
    # plt.plot(64, df[df['hidden_dim']==64]['auc'], marker='*', markersize=20, c='y')
plt.hlines(500, 1,1.5e3, 'k', linestyles='dashed', label='Oracle')
# plt.hlines(-10_000, 1,1e3, linestyles='dashed', label='Random')
plt.xlabel('Nodes in each hidden layer')
plt.ylabel('Area under eval reward curve')
plt.xscale('log')
plt.title(f'{env_id}')
# Put xlabel ticks in powers of 2:
plt.xticks([2**i for i in range(2, 11)], [2**i for i in range(2, 11)])
# tight layout:
plt.tight_layout()
plt.ylim(200,510)
plt.xlim(4,2**10)
plt.legend()
plt.savefig(f'{folder}/{env_id}.png')
plt.close()