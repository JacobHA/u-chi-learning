"""
make a combination of sweep configurations for all experiment and environment configurations.
Each environment has its default hyperparameters that will be added to the sweep configuration.
"""
import yaml, os


def update_sweep_cfg(sweepcfg, default_hparams):
    params = sweepcfg['parameters']
    for k, v in default_hparams.items():
        params[k] = {"values": [v]}
    sweepcfg['controller'] = {"type": "local"}
    return sweepcfg


def combine(experiment_name, env_id):
    default_params = yaml.safe_load(open(f'hparams/{env_id}/sql.yaml'))
    # load the sweep config
    with open(f"sweeps/{experiment_name}.yaml", "r") as f:
        expsweepcfg = yaml.load(f, yaml.SafeLoader)
    # set the default hyperparameters in the sweep config
    sweepcfg = update_sweep_cfg(expsweepcfg, default_params)
    filename = os.path.join('sweeps', f"{experiment_name}-{env_id}.yaml")
    with open(filename, "w") as f:
        yaml.dump(sweepcfg, f)
    print(f"saved {filename}")


if __name__ == "__main__":
    envs = os.listdir('hparams')
    sweeps = os.listdir('sweeps')
    for env in envs:
        for sweep in sweeps:
            # ignore if the already generated sweeps
            if any([x in sweep for x in envs]):
                continue
            try:
                combine(sweep.split('.')[0], env)
            except Exception as e:
                print(f"failed to combine {sweep} and {env}: {e}")