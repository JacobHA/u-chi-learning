import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob
from tbparse import SummaryReader

def clean_algorithm_name(algo_name, env):
    prefix = f'{env}-'
    if algo_name.startswith(prefix):
        return algo_name[len(prefix):]
    return algo_name


metrics_to_ylabel = {
    'eval/avg_reward': 'Average Evaluation Reward',
    'rollout/ep_reward': 'Average Rollout Reward',
    'train/theta': r'Reward-rate, $\theta$',
    'train/avg logu': r'Average of $\log u(s,a)$',
    'rollout/avg_entropy': r'Policy Entropy',
}
all_metrics = [
    'rollout/reward', 'eval/avg_reward', 'train/theta', 'train/avg logu'
]
sns.set_theme(style="whitegrid")
sns.set_context("poster")
plt.rcParams['text.color'] = 'black'

algo_to_color = {
    'SAC0.990.2': 'orange',
    'ASAC': 'blue',
    'arDDPG': 'green',
    'SQL': 'orange',
    'ASQL': 'blue',
    'DQN': 'green',
}


def plotter(env, folder, x_axis='step', metric='eval/avg_reward',
            exclude_algos=[], include_algos=[],
            xlim=None, ylim=None, ax=None):

    algo_data = pd.DataFrame()
    subfolders = glob(os.path.join(folder, '*'))
    print("Found subfolders:", subfolders)
    subfolders = sorted(subfolders)

    for subfolder in subfolders:
        if not os.path.isdir(subfolder) or subfolder.endswith('.png'):
            continue

        algo_name = os.path.basename(subfolder).split('_')[0]
        if algo_name in include_algos or len(include_algos) == 0:
            if algo_name in exclude_algos:
                print(f"Skipping {algo_name}, in exclude_algos.")
                continue
            log_files = glob(os.path.join(subfolder, '*.tfevents.*'))
            if not log_files:
                print(f"No log files found in {subfolder}")
                continue
            
            print("Processing", os.path.basename(subfolder))

            try:
                for log_file in log_files:
                    reader = SummaryReader(log_file)
                    df = reader.scalars
                    df = df[df['tag'].isin([metric, x_axis])]
                    clean_algo_name = clean_algorithm_name(algo_name, env)
                    
                    df['algo'] = clean_algo_name
                    df['run'] = os.path.basename(subfolder).split('_')[1]
                    algo_data = pd.concat([algo_data, df])
            except Exception as e:
                print(f"Error processing: {e}", log_file)
                continue

    metric_data = algo_data[algo_data['tag'] == metric]
    if not metric_data.empty:
        print(f"Plotting {metric}...")
        algo_runs = metric_data.groupby('algo')['run'].nunique()
        for algo, runs in algo_runs.items():
            sns.lineplot(data=metric_data[metric_data['algo']==algo], x='step', y='value', ax=ax, color=algo_to_color.get(algo, 'black'), label=algo)
            print(f"Plotted {algo}.")
        if metric == 'rollout/avg_entropy':
            ax.set_yscale('log')

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('Environment Steps')
        ax.set_ylabel(metrics_to_ylabel.get(metric, metric))
        ax.legend()

cc = ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0', 'LunarLander-v2']
mj = ['Pendulum-v1',  'Swimmer-v4', 'HalfCheetah-v4', 'Ant-v4']

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--envs', type=str, nargs='+', default=cc)
    parser.add_argument('-n', '--experiment_name', type=str, default='EVAL')
    args = parser.parse_args()
    envs = args.envs

    metrics = ['eval/avg_reward']
    for metric in metrics:
        fig, axis = plt.subplots(1, len(envs), figsize=(12*len(envs), 8))
        if len(envs) == 1:
            axis = [axis]
            env_name = envs[0]
        else:
            env_name = ''
        for i, env in enumerate(envs):
            ax = axis[i]
            ax.set_title(env)
            print(f"Plotting for {env} env.")
            folder = f'ft_logs/{args.experiment_name}/{env}'
            env_to_settings = {
                "Acrobot-v1": {
                    "xlim": (0, 10000),
                },
            }
            try:
                plotter(env=env,
                        folder=folder,
                        metric=metric,
                        exclude_algos=[],
                        **env_to_settings.get(env, {}),
                        include_algos=[],
                        ax=axis[i]
                        )
                if i == 0:
                    ax.set_ylabel(metrics_to_ylabel[metric])
                else:
                    ax.set_ylabel('')
            except KeyError:
                print("No data to plot.")
        
        unique_handles, unique_labels = [], []
        for ax in axis:
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                if label not in unique_labels:
                    unique_labels.append(label)
                    unique_handles.append(handle)
            ax.legend().remove()
        axis[0].legend(loc='upper left', ncol=1, borderaxespad=0., labels=unique_labels, handles=unique_handles)
        fig.tight_layout()
        save_path = os.path.join(f'ft_logs/{args.experiment_name}', env_name, f"{metric.split('/')[-1]}.png")
        print(f"Saving plot in {save_path}")
        plt.savefig(save_path, dpi=300)
        plt.close()
