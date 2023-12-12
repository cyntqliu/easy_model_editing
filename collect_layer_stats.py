from transformers import AutoModelForCausalLM, AutoTokenizer

from util.globals import MODELS, STATS_DIR
from util.nethook import set_requires_grad
from rome.layer_stats import layer_stats


def main():
    """
    Command-line utility to precompute cached stats.
    """
    import argparse

    parser = argparse.ArgumentParser(description="ROME Statistics Collector")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--model_name", default="gpt2-xl", choices=MODELS)
    aa("--dataset", default="wikipedia", choices=["wikitext", "wikipedia"])
    aa("--layers", default=[17], type=lambda x: list(map(int, x.split(","))))
    aa("--to_collect", default=["mom2"], type=lambda x: x.split(","))
    aa("--sample_size", default=100000, type=lambda x: None if x == "all" else int(x))
    aa("--batch_tokens", default=None, type=lambda x: None if x == "any" else int(x))
    aa("--precision", default="float32", choices=["float64", "float32", "float16"])
    aa("--stats_dir", default=STATS_DIR)
    aa("--download", default=1, type=int, choices=[0, 1])
    aa("--cpu", action="store_true")
    args = parser.parse_args()

    # sanity checking arguments
    if args.cpu:
        assert args.precision == "float32", (
            "Tensor operations for half and double precision are unavailable on CPU, please use float32 precision."
        )

    print(f"Loading {args.model_name} model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).eval()
    if not(args.cpu):
        model = model.cuda()
    set_requires_grad(False, model)
    print("Done.")

    for layer_num in args.layers:
        print(
            f"Computing stats for layer {layer_num} of {args.model_name} "
            f'over {args.sample_size or "all"} samples of {args.dataset}. '
            "Note, the statistics are collected over the inputs to the second MLP layer, "
            "or equivalently the outputs of the first MLP layer."
        )
        proj_layer_name = "c_proj" if "gpt2" in args.model_name else "fc_out"

        layer_name = f"transformer.h.{layer_num}.mlp.{proj_layer_name}"
        if args.model_name == "EleutherAI/gpt-neox-20b":
            # probably the layer name is bugged
            layer_name = f"gpt_neox.layers.{layer_num}.mlp.dense_4h_to_h"

        layer_stats(
            model,
            tokenizer,
            layer_name,
            args.stats_dir,
            args.dataset,
            args.to_collect,
            sample_size=args.sample_size,
            precision=args.precision,
            batch_tokens=args.batch_tokens,
            download=args.download,
            use_cpu=args.cpu,
        )

if __name__ == "__main__":
    main()
