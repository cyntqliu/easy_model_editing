This file explains how to run the experiments and evaluations

Note: Some of the python commands might need to be prepended by `PYTHONPATH=$PYTHONPATH:/path/to/the/repo/for/specificityplus`
(since specificityplus is currently not easily installable)


# Running MEMIT experiments

## Step 0: download git and setup environment
1) git clone this repo https://github.com/cyntqliu/easy_model_editing and cd to the repo
2) Setup conda enviroment: conda env create -f scripts/memit.yml
3) conda activate memit

## Step 1: download models + data
1) set ROOT_DIR by exporting an environment variable: export ROOT_DIR="/disk/scratch/s1785649/specificityplus/" or similar
2) python setup_data/download_data.py
3) python setup_data/download_hfdata.py
4) python setup_data/download_models.py --models gpt2-xl

## Step 2: precompute layer stats
1) If CUDA is available, run `python collect_layer_stats.py --model_name gpt2-xl --layers 8 --to_collect mom2 --sample_size 100000 --precision float32 --download 1`
2) Otherwise, add the `--cpu` flag and reduce the sample size

## Step 3: Run experiments
1) python run_eval.py --models gpt2-xl --algs MEMIT FT IDENTITY --split_into 125
2) Find lines to run in experiment-scripts/exp_gpt2xl.txt. Run one
3) Results for gpt2-xl are stored in results/

IF everything works for gpt2-xl, feel free to change the models by changing the --models flag in the `python setup_data/download_models.py` command, and rerunning steps 2 and 3 for said model.
