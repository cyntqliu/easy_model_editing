## run on headnode
import argparse

from transformers import AutoTokenizer
from huggingface_hub import snapshot_download


#get models
parser = argparse.ArgumentParser()
parser.add_argument(
    "--models",
    nargs='+',
    choices=["gpt2-medium", "gpt2-xl", "EleutherAI/gpt-j-6B", "EleutherAI/gpt-neox-20b"],
    required=True,
    help=f"Which models to download. Choices: ['gpt2-medium', 'gpt2-xl', 'EleutherAI/gpt-j-6B', 'EleutherAI/gpt-neox-20b'],"
)
args = parser.parse_args()


models = args.models
for model_name in models:
    print("Downloading model: ", model_name)
    snapshot_download(model_name,resume_download = True, ignore_patterns =["*.msgpack","*.h5","*.ot"])
    print("Downloading tokenizer: ", model_name)
    AutoTokenizer.from_pretrained(model_name,resume_download = True)
