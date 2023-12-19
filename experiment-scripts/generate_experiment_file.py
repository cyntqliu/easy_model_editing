#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--models",
    nargs='+',
    choices=["gpt2-medium", "gpt2-xl", "EleutherAI/gpt-j-6B", "EleutherAI/gpt-neox-20b"],
    required=True,
    help=f"Which models to edit. Choices: ['gpt2-medium', 'gpt2-xl', 'EleutherAI/gpt-j-6B', 'EleutherAI/gpt-neox-20b'],"
)
parser.add_argument(
    '--algs',
    nargs='+',
    choices=["ROME","FT","MEND","MEMIT","IDENTITY"],
    help=f"Which algorithms to evaluate. Choices: ['ROME', 'FT', 'MEND', 'MEMIT', 'IDENTITY']"
)
parser.add_argument(
    "--split_into",
    default=125,
    type=int,
)
args = parser.parse_args()


# generate experiment script calls
examples = 22000
split_into = args.split_into

models = args.models
ALGS = args.algs

hparams_ROME = {"gpt2-medium":"gpt2-medium.json", "gpt2-xl":"gpt2-xl.json", "EleutherAI/gpt-j-6B":"EleutherAI_gpt-j-6B.json", "EleutherAI/gpt-neox-20b":"EleutherAI_gpt-neox-20b.json"}
hparams_MEND = {"gpt2-xl":"gpt2-xl_CF.json", "EleutherAI/gpt-j-6B":"EleutherAI_gpt-j-6B_CF.json"}
hparams_IDENTITY = {"gpt2-medium":"identity_hparams.json", "gpt2-xl": "identity_hparams.json", "EleutherAI/gpt-j-6B":"identity_hparams.json", "EleutherAI/gpt-neox-20b":"identity_hparams.json"}
hparams_FT = {"gpt2-medium":"gpt2-medium_constr.json", "gpt2-xl":"gpt2-xl_constr.json", "EleutherAI/gpt-j-6B":"EleutherAI_gpt-j-6B_constr.json", "EleutherAI/gpt-neox-20b":"EleutherAI_gpt-neox-20b_constr.json"}
hparams_MEMIT = {"gpt2-xl":"gpt2-xl.json", "EleutherAI/gpt-j-6B":"EleutherAI_gpt-j-6B.json"}

hparams_dict = {"ROME":hparams_ROME, "MEND":hparams_MEND, "IDENTITY":hparams_IDENTITY, "FT":hparams_FT, "MEMIT":hparams_MEMIT}

filename = {"gpt2-medium":"exp_gpt2medium.txt", "gpt2-xl":"exp_gpt2xl.txt", "EleutherAI/gpt-j-6B":"exp_gptJ6B.txt", "EleutherAI/gpt-neox-20B":"exp_gptneox20b.txt"}

dataset_size = examples // split_into
if dataset_size * split_into != examples:
    raise ValueError("Dataset size must be divisible by split_into")

# get starting indexes
start_indexes = [i * examples // split_into for i in range(split_into)]

for model in models:
    file_path = "experiment-scripts/" + filename[model]
    output_file = open(file_path, "w")

    alg_names = " ".join(ALGS)
    hparams_fnames = [hparams_dict[alg][model] for alg in ALGS]
    hparams_fnames = " ".join(hparams_fnames)

    base_call = (f"python experiments/e2e.py --model_name {model} \
--alg_names {alg_names} --hparams_fnames {hparams_fnames} \
--ds_name cf --verbose --dataset_size_limit {dataset_size}")

    for start_i in start_indexes:
        # Note that we don't set a seed for rep - a seed is selected at random
        # and recorded in the output data by the python script
        expt_call = (
            f"{base_call} "
            f"--start_index {start_i} "
        )

        call = expt_call

        print(call, file=output_file)
    print (f"Wrote {len(start_indexes)} commands to {file_path}")
    output_file.close()

