import json

import pandas as pd
from tabulate import tabulate


def generate_latex_table(data, caption="Table", label="table", transpose=False):
    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(data)
    if transpose:
        df = df.T
    # Generate LaTeX table
    latex_table = tabulate(df, headers='keys', tablefmt='latex_booktabs', showindex=transpose)  # floatfmt=".2e"

    # Swap header and first row
    if transpose:
        lines = latex_table.split('\n')
        latex_table = '\n'.join(lines[0:2] + [lines[4], lines[3]] + lines[5:])

    # Add caption and label
    latex_table = f"\\begin{{table}}[ht]\n\\captionof{{table}}{{Finetuned Hyperparameter Values for {caption}}}\n\\centering\n{latex_table}\n\\label{{{label}}}\n\\end{{table}}"

    return latex_table


data_path = "best_hparams_tmp/hparams.json"
with open(data_path, 'r') as f:
    rawdata = json.load(f)
env_str_to_name = {
    "cpole": "CartPole-v1",
    "acro": "Acrobot-v1",
    "lunar": "LunarLander-v2",
    "mcar": "MountainCar-v0",
}
algo_to_name = {
    "sql": "SQL",
    "asql": "ASQL",
}
key_to_clean_key = {
    "batch\_size": "batch\ size",# "b",
    "aggregator": "agg",
    "hidden\_dim": "hidden\ dim",# "hid.\ dim",
    "learning\_rate": "learning\ rate",# "\\eta",
    "train\_freq": "train\ freq", # "TF"
    "gradient\_steps": "grad\ steps",
    "learning\_starts": "LS", #"learn\ starts",
    "tau\_theta": "\\tau_{\\theta}",
    "tau": "\\tau",
    "gamma": "\\gamma",
    "beta": "\\beta",
    "max\_grad\_norm": "MGN",
    "target\_update\_interval": "target\ update\ interval"# "TUI",
}
greek_letters = {"beta", "gamma", "alpha", "tau", "epsilon", "lambda", "rho", "sigma", "omega", "delta", "theta", "mu", "nu", "xi", "zeta", "kappa", "iota", "omicron", "upsilon", "phi", "chi", "psi", "omega"}
# make flat
target_algo = "sql"
data = []
for k, v in rawdata.items():
    algo, k_ = k.split("_")
    if algo != target_algo:
        continue
    val = {"Environment": env_str_to_name[k_]}
    for k2, v2 in v.items():
        if k2=='_wandb':
            continue
        val[k2] = v2.get('value', "-")
        if isinstance(val[k2], float):
            val[k2] = f"{val[k2]:.2e}"
    data.append(val)
# scan for columns with all the same values and remove them
single_val_cols = {}
for k in data[0].keys():
    vals = set([d[k] for d in data if k in d])
    if len(vals) == 1:
        single_val_cols[k] = vals.pop()
# drop columns
for k in single_val_cols.keys():
    for d in data:
        d.pop(k)

latex_table = generate_latex_table(data, caption=algo_to_name[target_algo], label=f"table:{target_algo}", transpose=True)

# post processing
# scienfic notation
# latex_table.replace("e-0", r"\times 10^{-1}")
import re
latex_table = re.sub(r"e-(\d+)", r"\\times 10^{-\1}", latex_table)
latex_table = re.sub(r"e+(\d+)", r"\\times 10^{+\1}", latex_table)
latex_table = latex_table.replace("e+00", "")
# remove leading zeros
latex_table = latex_table.replace("-0", "-")
latex_table = latex_table.replace("+0", "+")
# greek latters and other names
for k, v in key_to_clean_key.items():
    latex_table = latex_table.replace(k, v)

print(latex_table.replace("nan", "-"))

print(f"repeated value columns: \n{single_val_cols}")