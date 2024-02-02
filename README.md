EVAL: EigenVector-based Average-reward Learning

Welcome reviewers! Thank you for visiting the code associated to our ICML submission.
In this repository you will find (within darer) a BaseAgent class which we subclass for SQL and EVAL, as well as several helper functions. The hyperparameters listed in the appendix are stored in darer/hparams.py.

First install the requirements.txt file (e.g. with pip).

The darer/UAgent.py script is configured so that you may run it with

python darer/UAgent.py

to see results on Acrobot. 

The experiments/local_finetuned_runs.py file will run the experiments with finetuned hyperparameters.

Model-based ground truth comparisons with tabular algorithms:

![eigvec](figures/left_eigenvector_MB.png)
![policy](figures/policy_MB.png)

Model-free ground truth comparisons:

![eigvec][eigvec_figure]
![policy][policy_figure]

[policy_figure]: figures/policy_MF.png
[eigvec_figure]: figures/left_eigenvector_MF.png
