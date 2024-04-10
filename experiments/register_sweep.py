"""
requires network access to the wandb server to create a sweep
will create a new sweep for an experiment based on yaml sweep config
if either there is no local sweep id file for that experiment, or if the
saved sweep_id file contains a hash that doesn't match that of the current experiment config
will print out the saved sweep id
"""
import argparse
import hashlib
import json
import os
import time
from traceback import print_tb
import sys

sys.path.append('darer')
from utils import safe_open
import wandb


SEMAPHORE_FN = "creating_sweep"
SWEEP_ID_FN = "sweep_id"


def try_get_sweep_id(exp_name, project, offline=False):
    sweep_cfg_fn = f'sweeps/{exp_name}.yaml'
    sweep_id_fn = f'{SWEEP_ID_FN}_{exp_name}'
    sweep_config = safe_open(sweep_cfg_fn)
    if offline:
        print("will use a local controller type")
        sweep_config["controller"] = {'type': 'local'}
    current_cfg_hash = hashlib.sha256(json.dumps(sweep_config, sort_keys=True).encode()).hexdigest()
    # see if the sweep for the config already exists
    if os.path.exists(sweep_id_fn):
        with open(sweep_id_fn, "r") as f:
            saved_sweep_hash = f.readline()[:-1]
            saved_sweep_id = f.readline()
        if current_cfg_hash == saved_sweep_hash:
            print('continuing the existing sweep')
            return saved_sweep_id
    # otherwise create an updated sweep
    if os.path.exists(SEMAPHORE_FN):
        time.sleep(2)
        print("config is already being created")
    else:
        print('creating a new sweep config')
        try:
            with open(SEMAPHORE_FN, "w") as f:
                f.write("creating config")
            sweep_id = wandb.sweep(sweep_config, project=project)
            with open(sweep_id_fn, "w") as f:
                f.writelines([str(current_cfg_hash), '\n', sweep_id])
            print("sweep started")
        except Exception as e:
            print(e)
            print_tb(e.__traceback__)
            raise e
        finally:
            os.remove(SEMAPHORE_FN)
    with open(sweep_id_fn, "r") as f:
        f.readline()
        sweep_id = f.readline()
    return sweep_id


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--project', type=str, default='u-chi-learning')
    args.add_argument('--exp-name', type=str, default='EVAL-mean')
    args.add_argument('--offline', action='store_true')

    args = args.parse_args()
    sweep_id = try_get_sweep_id(exp_name=args.exp_name, project=args.project, offline=args.offline)
    print("Sweep ID:", sweep_id)