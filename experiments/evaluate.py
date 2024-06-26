import json
import shutil
import sys
from contextlib import redirect_stdout, nullcontext
from itertools import islice
from time import time, strftime
from typing import Tuple, Union

import torch
import seedbank
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from baselines.identity import IDENTITYHyperParams, apply_identity_to_model
from baselines.ft import FTHyperParams, apply_ft_to_model
from baselines.mend import MENDHyperParams, MendRewriteExecutor
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    MultiCounterFactDataset,
    get_tfidf_vectorizer,
)
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from memit import MEMITHyperParams, apply_memit_to_model
from rome import ROMEHyperParams, apply_rome_to_model
from util import nethook
from util.globals import *
from util.globals import DATASETS, ALGOS

ALG_DICT = {
    "IDENTITY": (IDENTITYHyperParams, apply_identity_to_model),
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
    "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
}

DS_DICT = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
}


def tprint(s, *args, **kwargs):
    print(strftime("%X %x") + " " + s, *args, **kwargs)


def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    skip_generation_tests: bool,
    generation_test_interval: int,
    conserve_memory: bool,
    dir_name: str,
    num_edits: int = 1,
    use_cache: bool = False,
    verbose: bool = False,
    start_index: int = 0
):
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # Determine run directory
    # Create new dir if not continuing from prev run OR prev run doesn't exist
    run_dir = get_run_dir(dir_name, model_name, start_index, dataset_size_limit)
    run_dir.mkdir(parents=True, exist_ok=True)
    tprint(f"Results will be stored at {run_dir}")

    # Get run hyperparameters
    params_path = HPARAMS_DIR / alg_name / hparams_fname
    shutil.copyfile(params_path, run_dir / "params.json")
    hparams = params_class.from_json(params_path)
    tprint(f"Executing {alg_name} with parameters {hparams}")

    # Instantiate vanilla model
    if type(model_name) is str:
        tprint(f"Instantiating model {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
        tprint(f"Model loaded to device: {model.device}")
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name
        model_name = model.config._name_or_path

    # Load data
    tprint("Loading dataset, attribute snippets, tf-idf data")
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None

    if num_edits > 1:
        assert ds_name != "cf", f"{ds_name} does not support multiple edits"

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, tok=tok, size=dataset_size_limit, start_index=start_index)

    # Get cache templates
    cache_template = None
    if use_cache:
        cache_template = (
            KV_DIR
            / f"{model_name.replace('/', '_')}_{alg_name}"
            / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
        )
        tprint(f"Will load cache from {cache_template}")

    # Iterate through dataset
    for record_chunks in tqdm(chunks(ds, num_edits), file=sys.stdout, total=int(len(ds)/num_edits)):
        case_result_template = str(run_dir / "{}_edits-case_{}.json")

        # Is the chunk already done?
        already_finished = True
        for record in record_chunks:
            if not Path(
                case_result_template.format(num_edits, record["case_id"])
            ).exists():
                already_finished = False
                break
        if already_finished:
            continue

        # Compute weight changes + record weights that changed
        case_ids = [record["case_id"] for record in record_chunks]
        args_conserve_memory = (
            dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
            if conserve_memory
            else dict()
        )
        etc_args = dict(cache_template=cache_template) if any(alg in alg_name for alg in ["ROME", "MEMIT"]) else dict()

        seedbank.initialize(SEED)
        start = time()
        with nullcontext() if verbose else redirect_stdout(sys.stderr):
            edited_model, weights_copy = apply_algo(
                model,
                tok,
                [
                    {"case_id": record["case_id"], **record["requested_rewrite"]}
                    for record in record_chunks
                ],
                hparams,
                copy=False,
                return_orig_weights=True,
                **args_conserve_memory,
                **etc_args,
            )
        exec_time = time() - start

        # Evaluate new model
        gen_test_vars = [snips, vec]
        for record in record_chunks:
            seedbank.initialize(SEED)
            out_file = Path(case_result_template.format(num_edits, record["case_id"]))
            if out_file.exists():
                tprint(f"Skipping {out_file}; already exists")
                continue

            metrics = {
                "case_id": record["case_id"],
                "grouped_case_ids": case_ids,
                "num_edits": num_edits,
                "model": model_name,
                "alg": alg_name,
                "requested_rewrite": record["requested_rewrite"],
                "time": exec_time,
                "post": ds_eval_method(
                    edited_model,
                    tok,
                    record,
                    *(
                        gen_test_vars
                        if record["case_id"] % generation_test_interval == 0
                        else [None, None]
                    ),  # Only test generation every generation_test_interval cases
                    out_file=out_file,
                ),
            }

            # Dump metrics in .json
            with open(out_file, "w") as f:
                json.dump(metrics, f, indent=1)

        # Restore original weights
        with torch.no_grad():
            for k, v in weights_copy.items():
                nethook.get_parameter(model, k)[...] = v.to("cuda")


def get_run_dir(dir_name: str, model_name: str, start_index: int, dataset_size_limit: int) -> Path:
    run_id = f"run_{str(start_index).zfill(5)}_{str(dataset_size_limit).zfill(5)}"
    return RESULTS_DIR / dir_name / model_name / run_id


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=ALGOS,
        default="ROME",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<model_name>/<run_id>, "
        "where the run_id is of the form 'run_<start_index>_<dataset_size_limit>'. ",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        choices=MODELS,
        default="gpt2-xl",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="gpt2-xl.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=DATASETS,
        default="mcf",
        help="Dataset to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to n records.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start index for the dataset. Useful for parallelizing runs.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=1,
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Also print detailed information about the running edit algorithm",
    )
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False, verbose=False)
    args = parser.parse_args()

    main(
        alg_name=args.alg_name,
        model_name=args.model_name,
        hparams_fname=args.hparams_fname,
        ds_name=args.ds_name,
        dataset_size_limit=args.dataset_size_limit,
        skip_generation_tests=args.skip_generation_tests,
        generation_test_interval=args.generation_test_interval,
        conserve_memory=args.conserve_memory,
        dir_name=args.alg_name,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
        verbose=args.verbose,
        start_index=args.start_index
    )


# helper:
# export PYTHONPATH=${PYTHONPATH}:~/git/specificityplus
# python experiments/evaluate.py --model_name=gpt2-xl --hparams_fname gpt2-xl_constr.json --alg_name IDENTITY --ds_name cf --start_index 20 --dataset_size_limit 10 --use_cache
