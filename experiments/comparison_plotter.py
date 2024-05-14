import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob
from tbparse import SummaryReader


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
# use poster settings:
sns.set_context("poster")
# Make font color black:
plt.rcParams['text.color'] = 'black'

algo_to_color = {
    'SAC0.990.2': 'orange',
    'ASAC': 'blue',
    'arDDPG': 'green',
}


def plotter(env, folder, x_axis='step', metric='eval/avg_reward',
            exclude_algos=[], include_algos=[],
            xlim=None, ylim=None, ax=None):

    algo_data = pd.DataFrame()
    subfolders = glob(os.path.join(folder, '*'))
    print("Found subfolders:", subfolders)
    # Sort the subfolders for consistent plotting colors (later can make a dict):
    subfolders = sorted(subfolders)
    # Collect all the data into one dataframe for parsing into figures:
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
            
            # Require only one log file per folder:
            # assert len(log_files) == 1
            print("Processing", os.path.basename(subfolder))

            try:
                for log_file in log_files:
                    reader = SummaryReader(log_file)
                    df = reader.scalars
                    df = df[df['tag'].isin(metrics + [x_axis])]
                    # Add a new column with the algo name:
                    clean_algo_name = algo_name.removeprefix(f'{env}-')
                    df['algo'] = clean_algo_name
                    # Add run number:
                    df['run'] = os.path.basename(subfolder).split('_')[1]
                    algo_data = pd.concat([algo_data, df])
            except Exception as e:
                print(f"Error processing: {e}", log_file)
                continue

    # Filter the data to only include this metric:
    metric_data = algo_data[algo_data['tag'] == metric]
    if not metric_data.empty:
        print(f"Plotting {metric}...")
        # Append the number of runs to the legend for each algo:
        algo_runs = metric_data.groupby('algo')['run'].nunique()
        for algo, runs in algo_runs.items():
            metric_data.loc[metric_data['algo'] == algo, 'algo'] = f"{algo}"  #  f"{algo} ({runs} runs)"
            sns.lineplot(data=metric_data[metric_data['algo']==algo], x='step', y='value', ax=ax, color=algo_to_color[algo], label=algo)  # hue='algo',
        if metric == 'rollout/avg_entropy':
            ax.set_yscale('log')

        try:
            name = metrics_to_ylabel[metric]
        except KeyError:
            print(f"metric {metric} not in metrics_to_ylabel dict.")
        ax.legend()
        # # strip the title from the values in legend:
        # handles, labels = ax.get_legend_handles_labels()
        # labels = []
        # for handle in handles:
        #     label = handle.get_label()
        #     try:
        #         labels.append(label.split(env+'-')[-1])
        #     except TypeError:
        #         labels.append(label)
        #     # swap U for EVAL:
        #     labels = [label.replace('U', 'EVAL') for label in labels]
        #     labels = [label.replace('ppi', 'PPI') for label in labels]
        #     # Remove the number of runs:
        #     # labels = [label.split(' (')[0] for label in labels]
        # # labels = [label.split(title+'-')[-1] for label in labels]
        # ax.legend(handles=handles, labels=labels)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('Environment Steps')

        # plt.tight_layout()

    else:
        print("No data to plot.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--envs', type=str, nargs='+', default=['Swimmer-v4', 'Ant-v4', 'HalfCheetah-v4', 'Swimmer-v4'])
    parser.add_argument('-n', '--experiment_name', type=str, default='ft_logs_test')
    args = parser.parse_args()
    envs = args.envs
    # envs = ['Acrobot-v1', 'CartPole-v1', 'MountainCar-v0', 'LunarLander-v2']

    metrics = ['eval/avg_reward',]  #  'eval/avg_reward', 'train/theta', 'rollout/neg_free_energy'
    for metric in metrics:
        fig, axis = plt.subplots(1, len(envs), figsize=(12*len(envs), 8))
        for i, env in enumerate(envs):
            ax = axis[i]
            ax.set_title(env)
            print(f"Plotting for {env} env.")
            # folder = f'experiments/ft/{env}/'
            folder = f'ft_logs/{args.experiment_name}/{env}'
            # folder = f'experiments/ablations/{env}/'
            env_to_settings = {
                "Acrobot-v1": {
                    "xlim": (0, 5000),
                    # "ylim": (-500, 0),
                },
            }
            try:
                plotter(env=env,
                        folder=folder,
                        metric=metric,
                        exclude_algos=[],  # f'{env}-arSAC-autoauto', f'{env}-arSAC-min', f'{env}-arSAC-autonewhauto', f'{env}-arSAC'
                        **env_to_settings.get(env, {}),
                        include_algos=[],  # f'{env}-arDDPG', 'SAC', f'{env}-arSAC-newh'
                        ax=axis[i]
                        )
                if i==0:
                    ax.set_ylabel(metrics_to_ylabel[metric])
                else:
                    ax.set_ylabel('')
            except KeyError:
                print("No data to plot.")
        # remove duplicate legend entries
        unique_handles, unique_labels = [], []
        for ax in axis:
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                if label not in unique_labels:
                    unique_labels.append(label)
                    unique_handles.append(handle)
            ax.legend().remove()
        # Put legend under the plot outside:
        axis[0].legend(loc='upper left', ncol=1, borderaxespad=0., labels=unique_labels, handles=unique_handles)
        fig.tight_layout()
        save_path = os.path.join(f'ft_logs/{args.experiment_name}', f"{metric.split('/')[-1]}.png")
        print(f"Saving plot in {save_path}")
        plt.savefig(save_path, dpi=300)
        plt.close()

        # Run from u-chi-learning directory: "python experiments/comparison_plotter.py -e ..."