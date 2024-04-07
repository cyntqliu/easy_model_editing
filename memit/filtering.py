'''
Code for filtering out superfluous edit requests
'''
from typing import Any, Dict, List

import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer

from util.utils import chunks


def semantically_equivalent(string_1: str, string_2: str) -> bool:
    '''
    Determines if `string_1` and `string_2` are semantically equivalent

    returns: bool indicating if `string_1` and `string_2` are equivalent
    '''
    # TODO: add semantics-based matching
    return string_1 == string_2


def _filter_requests_hf(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, requests: List[Dict]) -> List[Dict]:
    '''HuggingFace model implementation of `filter_requests`'''

    # changes to enable batch generation
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    filtered_requests = []
    for request_batch in chunks(requests, 32):
        inputs = [
            rq["requested_rewrite"]["prompt"].format(rq["requested_rewrite"]["subject"])
            for rq in request_batch
        ]
        inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")

        targets_greedy_raw = model.generate(**inputs, num_beams=1, do_sample=False)
        targets_greedy = tokenizer.batch_decode(targets_greedy_raw, skip_special_tokens=True)

        # generation may produce more text than just one sentence. TODO: Truncate
        # the outputs appropriately

        for output, request in zip(targets_greedy, requests):
            rq_target = request["requested_rewrite"]["target_new"]
            if not(semantically_equivalent(output, rq_target)):
                filtered_requests.append(request)

    return filtered_requests


def _filter_requests_pt(model: nn.Module, tokenizer: PreTrainedTokenizer, requests: List[Dict]) -> List[Dict]:
    pass


def filter_requests(model: Any, tokenizer: Any, requests: List[Dict]) -> List[Dict]:
    '''
    Intuitively, we only want to take requests whose facts don't already exist
    in the model. MEMIT uses output probability as a heuristic for some fact
    "existing." As a result, we say that a relation already exists within the
    model if its `target_new` is semantically equivalent to `target_greedy`,
    where `target_greedy` is found as follows:

    `target_greedy = model.generate(
        **inputs,
        num_beams=1,
        do_sample=False,
    )`

    Note: as of Mar 2024, the arguments above are also the default arguments
    used by HuggingFace generation.

    Based on the above definition, this method filters out any requests in `requests`
    that already exists within the model.

    Args:
        model: The target model to edit
        tokenizer: The tokenizer for use with the model. Must have the same API as
            HuggingFace PreTrainedTokenizer
        requests: List[Dict] where each Dict matches the request format specified
            in the `eme.get_request` method

    returns: List[Dict] consisting of a subset of `requests`
    '''

    if isinstance(model, nn.Module):
        return _filter_requests_pt(model, tokenizer, requests)
    if isinstance(model, PreTrainedModel):
        assert isinstance(tokenizer, PreTrainedTokenizer), (
            f"""Using a HuggingFace model without a HuggingFace tokenizer (got {type(tokenizer)})
            . Please pass in the model's associated Tokenizer."""
        )
        return _filter_requests_hf(model, tokenizer, requests)

    raise TypeError(
        f"""Got model of type {type(model)}, only PyTorch models (nn.Module) and
         HF models (PreTrainedModel) are supported."""
    )
