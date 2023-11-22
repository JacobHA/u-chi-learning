import random

import numpy as np
import wandb
import argparse
import sys
sys.path.append('darer')

from darer.MultiLogU import main


default_config = {
    "method": "random",
    "parameters": {
        'env_id': {"values": ['ALE/Pong-v5']}
    }
}

atari_v0 = {
    "metric": {
        "name": "eval/avg_reward",
        "goal": "maximize"
    },
    "parameters": {
        "env_id": {'values': [
            'ALE/Adventure-v5', 'ALE/Adventure-ram-v5', 'ALE/AirRaid-v5', 'ALE/AirRaid-ram-v5', 'ALE/Alien-v5', 'ALE/Alien-ram-v5', 'ALE/Amidar-v5', 'ALE/Amidar-ram-v5', 'ALE/Assault-v5', 'ALE/Assault-ram-v5', 'ALE/Asterix-v5', 'ALE/Asterix-ram-v5', 'ALE/Asteroids-v5', 'ALE/Asteroids-ram-v5', 'ALE/Atlantis-v5', 'ALE/Atlantis-ram-v5', 'ALE/Atlantis2-v5', 'ALE/Atlantis2-ram-v5',
'ALE/Backgammon-v5', 'ALE/Backgammon-ram-v5', 'ALE/BankHeist-v5', 'ALE/BankHeist-ram-v5', 'ALE/BasicMath-v5', 'ALE/BasicMath-ram-v5', 'ALE/BattleZone-v5', 'ALE/BattleZone-ram-v5', 'ALE/BeamRider-v5', 'ALE/BeamRider-ram-v5', 'ALE/Berzerk-v5', 'ALE/Berzerk-ram-v5', 'ALE/Blackjack-v5', 'ALE/Blackjack-ram-v5', 'ALE/Bowling-v5', 'ALE/Bowling-ram-v5', 'ALE/Boxing-v5', 'ALE/Boxing-ram-v5', 'ALE/Breakout-v5', 'ALE/Breakout-ram-v5', 'ALE/Carnival-v5', 'ALE/Carnival-ram-v5', 'ALE/Casino-v5', 'ALE/Casino-ram-v5', 'ALE/Centipede-v5', 'ALE/Centipede-ram-v5', 'ALE/ChopperCommand-v5', 'ALE/ChopperCommand-ram-v5', 'ALE/CrazyClimber-v5', 'ALE/CrazyClimber-ram-v5', 'ALE/Crossbow-v5', 'ALE/Crossbow-ram-v5', 'ALE/Darkchambers-v5', 'ALE/Darkchambers-ram-v5', 'ALE/Defender-v5', 'ALE/Defender-ram-v5', 'ALE/DemonAttack-v5', 'ALE/DemonAttack-ram-v5', 'ALE/DonkeyKong-v5', 'ALE/DonkeyKong-ram-v5', 'ALE/DoubleDunk-v5', 'ALE/DoubleDunk-ram-v5', 'ALE/Earthworld-v5', 'ALE/Earthworld-ram-v5', 'ALE/ElevatorAction-v5', 'ALE/ElevatorAction-ram-v5', 'ALE/Enduro-v5', 'ALE/Enduro-ram-v5', 'ALE/Entombed-v5', 'ALE/Entombed-ram-v5', 'ALE/Et-v5', 'ALE/Et-ram-v5', 'ALE/FishingDerby-v5', 'ALE/FishingDerby-ram-v5', 'ALE/FlagCapture-v5', 'ALE/FlagCapture-ram-v5', 'ALE/Freeway-v5', 'ALE/Freeway-ram-v5', 'ALE/Frogger-v5', 'ALE/Frogger-ram-v5', 'ALE/Frostbite-v5', 'ALE/Frostbite-ram-v5', 'ALE/Galaxian-v5', 'ALE/Galaxian-ram-v5', 'ALE/Gopher-v5', 'ALE/Gopher-ram-v5', 'ALE/Gravitar-v5', 'ALE/Gravitar-ram-v5', 'ALE/Hangman-v5', 'ALE/Hangman-ram-v5', 'ALE/HauntedHouse-v5', 'ALE/HauntedHouse-ram-v5', 'ALE/Hero-v5', 'ALE/Hero-ram-v5', 'ALE/HumanCannonball-v5', 'ALE/HumanCannonball-ram-v5', 'ALE/IceHockey-v5', 'ALE/IceHockey-ram-v5', 'ALE/Jamesbond-v5', 'ALE/Jamesbond-ram-v5', 'ALE/JourneyEscape-v5', 'ALE/JourneyEscape-ram-v5', 'ALE/Kaboom-v5', 'ALE/Kaboom-ram-v5', 'ALE/Kangaroo-v5', 'ALE/Kangaroo-ram-v5', 'ALE/KeystoneKapers-v5', 'ALE/KeystoneKapers-ram-v5', 'ALE/KingKong-v5', 'ALE/KingKong-ram-v5', 'ALE/Klax-v5', 'ALE/Klax-ram-v5', 'ALE/Koolaid-v5', 'ALE/Koolaid-ram-v5', 'ALE/Krull-v5', 'ALE/Krull-ram-v5', 'ALE/KungFuMaster-v5', 'ALE/KungFuMaster-ram-v5', 'ALE/LaserGates-v5', 'ALE/LaserGates-ram-v5', 'ALE/LostLuggage-v5', 'ALE/LostLuggage-ram-v5', 'ALE/MarioBros-v5', 'ALE/MarioBros-ram-v5', 'ALE/MiniatureGolf-v5', 'ALE/MiniatureGolf-ram-v5', 'ALE/MontezumaRevenge-v5', 'ALE/MontezumaRevenge-ram-v5', 'ALE/MrDo-v5', 'ALE/MrDo-ram-v5', 'ALE/MsPacman-v5', 'ALE/MsPacman-ram-v5', 'ALE/NameThisGame-v5', 'ALE/NameThisGame-ram-v5', 'ALE/Othello-v5', 'ALE/Othello-ram-v5', 'ALE/Pacman-v5', 'ALE/Pacman-ram-v5', 'ALE/Phoenix-v5', 'ALE/Phoenix-ram-v5', 'ALE/Pitfall-v5', 'ALE/Pitfall-ram-v5', 'ALE/Pitfall2-v5', 'ALE/Pitfall2-ram-v5', 'ALE/Pong-v5', 'ALE/Pong-ram-v5', 'ALE/Pooyan-v5', 'ALE/Pooyan-ram-v5', 'ALE/PrivateEye-v5', 'ALE/PrivateEye-ram-v5', 'ALE/Qbert-v5', 'ALE/Qbert-ram-v5', 'ALE/Riverraid-v5', 'ALE/Riverraid-ram-v5', 'ALE/RoadRunner-v5', 'ALE/RoadRunner-ram-v5', 'ALE/Robotank-v5', 'ALE/Robotank-ram-v5', 'ALE/Seaquest-v5', 'ALE/Seaquest-ram-v5', 'ALE/SirLancelot-v5', 'ALE/SirLancelot-ram-v5', 'ALE/Skiing-v5', 'ALE/Skiing-ram-v5', 'ALE/Solaris-v5', 'ALE/Solaris-ram-v5', 'ALE/SpaceInvaders-v5', 'ALE/SpaceInvaders-ram-v5', 'ALE/SpaceWar-v5', 'ALE/SpaceWar-ram-v5', 'ALE/StarGunner-v5', 'ALE/StarGunner-ram-v5', 'ALE/Superman-v5', 'ALE/Superman-ram-v5', 'ALE/Surround-v5', 'ALE/Surround-ram-v5', 'ALE/Tennis-v5', 'ALE/Tennis-ram-v5', 'ALE/Tetris-v5', 'ALE/Tetris-ram-v5', 'ALE/TicTacToe3D-v5', 'ALE/TicTacToe3D-ram-v5', 'ALE/TimePilot-v5', 'ALE/TimePilot-ram-v5', 'ALE/Trondead-v5', 'ALE/Trondead-ram-v5', 'ALE/Turmoil-v5', 'ALE/Turmoil-ram-v5', 'ALE/Tutankham-v5', 'ALE/Tutankham-ram-v5', 'ALE/UpNDown-v5', 'ALE/UpNDown-ram-v5', 'ALE/Venture-v5', 'ALE/Venture-ram-v5', 'ALE/VideoCheckers-v5', 'ALE/VideoCheckers-ram-v5', 'ALE/VideoChess-v5', 'ALE/VideoChess-ram-v5', 'ALE/VideoCube-v5', 'ALE/VideoCube-ram-v5', 'ALE/VideoPinball-v5', 'ALE/VideoPinball-ram-v5', 'ALE/WizardOfWor-v5', 'ALE/WizardOfWor-ram-v5', 'ALE/WordZapper-v5', 'ALE/WordZapper-ram-v5', 'ALE/YarsRevenge-v5', 'ALE/YarsRevenge-ram-v5', 'ALE/Zaxxon-v5', 'ALE/Zaxxon-ram-v5'
            ]
        }
    }
}

exp_to_config = {
    "atari-v0": atari_v0
}

def sample_wandb_hyperparams(params):
    sampled = {}
    for k, v in params.items():
        if 'values' in v:
            sampled[k] = random.choice(v['values'])
        elif 'distribution' in v:
            if v['distribution'] == 'uniform' or v['distribution'] == 'uniform_values':
                sampled[k] = random.uniform(v['min'], v['max'])
            elif v['distribution'] == 'normal':
                sampled[k] = random.normalvariate(v['mean'], v['std'])
            elif v['distribution'] == 'log_uniform_values':
                emin, emax = np.log(v['max']), np.log(v['min'])
                sample = np.exp(random.uniform(emin, emax))
                sampled[k] = sample
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    return sampled


def get_sweep_config(sweepcfg, default_config, project_name):
    cfg = default_config
    params = cfg['parameters']
    params.update(sweepcfg['parameters'])
    cfg.update(sweepcfg)
    cfg['parameters'] = params
    cfg['name'] = project_name
    return cfg


def wandb_train(local_cfg=None):
    """:param local_cfg: pass config sweep if running locally to sample without wandb"""
    wandb_kwargs = {"project":project, "group":experiment_name}
    if local_cfg:
        local_cfg["controller"] = {'type': 'local'}
        sampled_params = sample_wandb_hyperparams(local_cfg["parameters"])
        local_cfg["parameters"] = sampled_params
        print(f"locally sampled params: {sampled_params}")
        wandb_kwargs['config'] = local_cfg
    with wandb.init(**wandb_kwargs, sync_tensorboard=True) as run:
        config = wandb.config.as_dict()
        main(**config['parameters'])


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--sweep", type=str, default=None)
    args.add_argument("--n_runs", type=int, default=1)
    args.add_argument("--proj", type=str, default="u-chi-learning")
    args.add_argument("--local-wandb", type=bool, default=True)
    args.add_argument("--exp-name", type=str, default="atari-v0")
    args = args.parse_args()
    project = args.proj
    experiment_name = args.exp_name
    expsweepcfg = exp_to_config[experiment_name]
    sweepcfg = get_sweep_config(expsweepcfg, default_config, project)
    if args.sweep is None and not args.local_wandb:
        sweep_id = wandb.sweep(sweepcfg, project=project)
        print(f"created new sweep {sweep_id}")
        wandb.agent(sweep_id, project=args.proj, count=args.n_runs, function=wandb_train)
    elif args.local_wandb:
        wandb_train(local_cfg=sweepcfg)
    else:
        print(f"continuing sweep {args.sweep}")
        wandb.agent(args.sweep, project=args.proj, count=args.n_runs, function=wandb_train)
