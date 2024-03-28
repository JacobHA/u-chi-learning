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

def plotter(env, folder, x_axis='step', metrics=all_metrics, exclude_algos=[],
            xlim=None, ylim=None, title=None):

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
                df['algo'] = algo_name
                # Add run number:
                df['run'] = os.path.basename(subfolder).split('_')[1]
                algo_data = pd.concat([algo_data, df])
        except Exception as e:
            print(f"Error processing: {e}", log_file)
            continue

    # Now, loop over all the metrics and plot them individually:
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        plt.title(title)
        # Filter the data to only include this metric:
        metric_data = algo_data[algo_data['tag'] == metric]
        if not metric_data.empty:
            print(f"Plotting {metric}...")
            # Append the number of runs to the legend for each algo:
            algo_runs = metric_data.groupby('algo')['run'].nunique()
            for algo, runs in algo_runs.items():
                metric_data.loc[metric_data['algo'] == algo, 'algo'] = f"{algo} ({runs} runs)"
            sns.lineplot(data=metric_data, x='step', y='value', hue='algo')
            if metric == 'rollout/avg_entropy': 
                plt.yscale('log')

            try:
                name = metrics_to_ylabel[metric]
            except KeyError:
                print("Add metric to metrics_to_ylabel dict.")
            
            # Put legend under the plot outside:
            plt.legend(loc='lower right', ncol=1, borderaxespad=0.)
            # strip the title from the values in legend:
            handles, labels = plt.gca().get_legend_handles_labels()
            labels = []
            for handle in handles:
                label = handle.get_label()
                try:
                    labels.append(label.split(env+'-')[-1])
                except TypeError:
                    labels.append(label)
                # swap U for EVAL:
                labels = [label.replace('U', 'EVAL') for label in labels]
                # Swap Rawlik for PPI:
                labels = [label.replace('rawlik', 'PPI') for label in labels]
                # Remove the number of runs:
                # labels = [label.split(' (')[0] for label in labels]
            # labels = [label.split(title+'-')[-1] for label in labels]
            plt.gca().legend(handles=handles, labels=labels)
                
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.xlabel('Environment Steps')
            plt.ylabel(name)

            plt.tight_layout()
            plt.savefig(os.path.join(folder, f"{metric.split('/')[-1]}-{env}.png"), dpi=300)
            plt.close()
        else:
            print("No data to plot.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env', type=str, default='')
    args = parser.parse_args()
    env = args.env
    if env == '':
        envs = ['Acrobot-v1', 'CartPole-v1', 'MountainCar-v0', 'LunarLander-v2']
    else:
        envs = [env]

    for env in envs:
        print(f"Plotting for {env} env.")
        # folder = f'experiments/ft/{env}/'
        folder = f'ft_logs/{env}'
        # folder = f'experiments/ablations/{env}/'
        # env_to_settings = {
        

        # plotter(folder=folder, metrics=['eval/avg_reward'], ylim=(0, 510), exclude_algos=['CartPole-v1-U','CartPole-v1-Umin',  'CartPole-v1-Ured', 'CartPole-v1-Umean', 'CartPole-v1-Umse-b02', ])
        # plotter(folder=folder, metrics=['rollout/ep_reward'], ylim=(0, 510), exclude_algos=['CartPole-v1-U','CartPole-v1-Umin', 'CartPole-v1-Ured', 'CartPole-v1-Umean', 'CartPole-v1-Umse-b02', ])

        try:
            plotter(env=env, folder=folder, metrics=['eval/avg_reward'], title=env)
        except KeyError:
            print("No data to plot.")

        # Run from u-chi-learning directory: "python experiments/comparison_plotter.py -e ..."