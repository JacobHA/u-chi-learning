import wandb
import argparse
import yaml

with open("sweeps/EVAL-mean.yaml") as f:
    yaml_contents = yaml.load(f, yaml.FullLoader)
yaml_contents["controller"] = {'type': 'local'}

argparse = argparse.ArgumentParser()
argparse.add_argument('--sweep_id', type=str, default=None)
argparse.add_argument('--project', type=str, default='u-chi-learning')
argparse.add_argument('--entity', type=str, default=None)
args = argparse.parse_args()
sweep_controller = wandb.controller(yaml_contents, project=args.project, entity=args.entity)
sweep_controller.run()