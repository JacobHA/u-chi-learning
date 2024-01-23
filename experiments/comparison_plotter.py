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
}
all_metrics = [
    'rollout/reward', 'eval/avg_reward', 'train/theta', 'train/avg logu'
]
sns.set_theme(style="darkgrid")

def plotter(folder, x_axis='step', metrics=all_metrics, exclude_algos=[],
            xlim=None, ylim=None):

    algo_data = pd.DataFrame()
    subfolders = glob(os.path.join(folder, '*'))
    print("Found subfolders:", subfolders)

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
        assert len(log_files) == 1
        log_file = log_files[0]
        print("Processing", os.path.basename(subfolder))

        try:
            reader = SummaryReader(log_file)
            df = reader.scalars
            df = df[df['tag'].isin(metrics + [x_axis])]
            # Add a new column with the algo name:
            df['algo'] = algo_name
            # Add run number:
            df['run'] = os.path.basename(subfolder).split('_')[1]
            algo_data = pd.concat([algo_data, df])
        except Exception as e:
            print("Error processing", log_file)
            continue

    # Now, loop over all the metrics and plot them individually:
    for metric in metrics:
        plt.figure()
        # Filter the data to only include this metric:
        metric_data = algo_data[algo_data['tag'] == metric]
        if not metric_data.empty:
            print(f"Plotting {metric}...")
            # Append the number of runs to the legend for each algo:
            algo_runs = metric_data.groupby('algo')['run'].nunique()
            for algo, runs in algo_runs.items():
                metric_data.loc[metric_data['algo'] == algo, 'algo'] = f"{algo} ({runs} runs)"
            sns.lineplot(data=metric_data, x='step', y='value', hue='algo')
            try:
                name = metrics_to_ylabel[metric]
            except KeyError:
                print("Add metric to metrics_to_ylabel dict.")
            
            plt.legend()

            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.xlabel('Environment Steps')
            plt.ylabel(name)

            plt.savefig(os.path.join(folder, f"{metric.split('/')[-1]}.png"))
            plt.close()
        else:
            print("No data to plot.")

if __name__ == "__main__":
    folder = 'experiments/ft/Acrobot-v1'
    # folder = 'experiments/ft/CartPole-v1'
    folder = 'experiments/ft/MountainCar-v0'

    # plotter(folder=folder, metrics=['eval/avg_reward'], ylim=(0, 510), exclude_algos=['CartPole-v1-U','CartPole-v1-Umin',  'CartPole-v1-Ured', 'CartPole-v1-Umean', 'CartPole-v1-Umse-b02', ])
    # plotter(folder=folder, metrics=['rollout/ep_reward'], ylim=(0, 510), exclude_algos=['CartPole-v1-U','CartPole-v1-Umin', 'CartPole-v1-Ured', 'CartPole-v1-Umean', 'CartPole-v1-Umse-b02', ])

    plotter(folder=folder, metrics=['eval/avg_reward'])
    plotter(folder=folder, metrics=['rollout/ep_reward'])

    # plotter(folder=folder, metrics=['step', 'train/theta', 'theta'])
    # plotter(folder=folder, metrics=['step', 'train/avg logu', 'avg logu'])