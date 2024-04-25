"""
extracts best hyperparameters from wandb project for each group
"""
import os

import yaml
import argparse
import wandb


def extract_best_hparams(entity_name, project_name, discrete_hyperparameter, env_id, optimizing_metric, out_folder):
    # Fetch runs from the specified project
    api = wandb.Api()
    runs = api.runs(f"{entity_name}/{project_name}")
    # Filter runs by the discrete hyperparameter and find the best one based on the optimizing metric
    best_hyperparams_per_value = {}
    for run in runs:
        # Extract run data
        config = run.config
        summary = run.summary

        # Check if the discrete hyperparameter is in this run's config
        if summary.get("env_id")==env_id and discrete_hyperparameter in config:
            hyperparam_value = config[discrete_hyperparameter]
            metric_value = summary.get(optimizing_metric)
            if metric_value is None:
                continue

            # Update the best hyperparameters for this value
            if hyperparam_value not in best_hyperparams_per_value or \
                    best_hyperparams_per_value[hyperparam_value]['best_metric'] < metric_value:
                best_hyperparams_per_value[hyperparam_value] = {
                    'best_metric': metric_value,
                    'hyperparameters': config
                }

    # Print the best hyperparameters for each value of the specified hyperparameter
    # make output dir if doesn't exist
    os.makedirs(out_folder, exist_ok=True)
    for value, data in best_hyperparams_per_value.items():
        print(f"Value: {value}, Best {optimizing_metric}: {data['best_metric']}, Hyperparameters: {data['hyperparameters']}")
        # export best hyperparameters to a yaml file
        with open(f"{out_folder}/{env_id}/{data['hyperparameters'][discrete_hyperparameter]}.yaml", 'w') as file:
            if data['hyperparameters'][discrete_hyperparameter] == 'none':
                yaml.dump(data['hyperparameters'], file)
            else:
                yaml.dump({discrete_hyperparameter: data['hyperparameters'][discrete_hyperparameter]}, file)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--entity_name', type=str, default=None)
    args.add_argument('--project_name', type=str, default='clipping')
    args.add_argument('--discrete_hyperparameter', type=str, default='algo_name')
    args.add_argument('--env_id', type=str, default='LunarLander-v2')
    args.add_argument('--optimizing_metric', type=str, default='eval/auc')
    args.add_argument('--out-folder', type=str, default='./hparams')
    args = args.parse_args()
    extract_best_hparams(args.entity_name, args.project_name, args.discrete_hyperparameter, args.env_id, args.optimizing_metric, args.out_folder)