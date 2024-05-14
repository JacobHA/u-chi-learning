Hello Reviewers! Thank you for taking the time to review our work and visit the code. We have provided a brief overview of the contents of this repository below. If you have any questions or need further information, please do not hesitate to reach out to us.

# Overview
- darer (orig. Deep Average Reward w Entropy Regularization)
Contains the code for our algorithms (ASQL.py, ASAC.py) and the baselines (arDDPG.py, CustomDQN.py, CustomSAC.py) which are built on top of a BaseAgent class and other utils. For Atari experiments, we required our own wrappers for the gymnasium version of the environments, which can be found in "wrappers.py".

Finally, Models.py contains the (standard MLP or CNN) neural network architectures used in our experiments.

Much of the code is built with the stable-baselines3 style in mind, and the CustomDQN and CustomSAC files are simply subclassing the corresponding agents. This allows us to log consistent metrics and run sweeps & finetuned runs similarly.

- ASQL:
As shown in the pseudocode in the main text, ASQL closely follows a DQN/SQL style implementation. The relevant changes are tracking the theta value and using the correct TD update equation for the target.

- ASAC:
Again, ASAC closely follows a SAC style implementation. The relevant changes are tracking the theta value and using the correct TD update equation for the target.


# Getting Started:
First to install the requirements (in a new conda environment):
```bash
pip install -r requirements.txt
```
Then, to run a finetuned (selected hparams) experiment:
```bash
python experiments/finetuned_runs.py --env Acrobot-v1 --algo ASQL
```

To run a sweep (over a range of hparams):
```bash
python experiments/sweep.py --env LunarLander-v2 --algo ASQL --project project_name --exp-name expt_name
```

The project_name will correspond to the wandb project name, and the expt_name will correspond to the sweep name. The sweep will be logged to wandb, and the results can be viewed there.

The different experiment sweep files can be found in the sweeps directory. We have used the EVAL sweep for our ASQL results, and mujoco for our ASAC results.