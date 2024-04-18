"""
Code for filtering out superfluous edit requests
"""

from typing import Any, Callable, Dict, List, Optional

import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer

from util.utils import chunks


def semantically_equivalent(string_1: str, string_2: str) -> bool:
    """
    Determines if `string_1` and `string_2` are semantically equivalent

    returns: bool indicating if `string_1` and `string_2` are equivalent
    """
    # TODO: add semantics-based matching
    return string_1 == string_2


def _filter_requests_hf(
    requests: List[Dict],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
) -> List[Dict]:
    """HuggingFace model implementation of `filter_requests`"""

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
        targets_greedy = tokenizer.batch_decode(
            targets_greedy_raw, skip_special_tokens=True
        )

        # generation may produce more text than just one sentence. TODO: Truncate
        # the outputs appropriately

        for output, request in zip(targets_greedy, requests):
            rq_target = request["requested_rewrite"]["target_new"]
            if not semantically_equivalent(output, rq_target):
                filtered_requests.append(request)

    return filtered_requests


def _filter_requests_pt(
    requests: List[Dict],
    model: Optional[nn.Module] = None,
    tokenizer: Any = None,
    model_forward: Optional[Callable] = None,
    tokenizer_encode: Optional[Callable] = None,
    tokenizer_decode: Optional[Callable] = None,
    model_forward_kwargs: Optional[Dict] = {},
    encode_kwargs: Optional[Dict] = {},
    decode_kwargs: Optional[Dict] = {},
) -> List[Dict]:
    """PyTorch model implementtation of `filter_requests`"""

    filtered_requests = []
    for request_batch in chunks(requests, 32):
        inputs = [
            rq["requested_rewrite"]["prompt"].format(rq["requested_rewrite"]["subject"])
            for rq in request_batch
        ]
        if tokenizer:
            inputs = tokenizer(
                inputs, padding=True, truncation=True, return_tensors="pt"
            )
        else:
            inputs = tokenizer_encode(inputs, **encode_kwargs)

        if model:
            targets_greedy_raw = model(inputs, **model_forward_kwargs)
        else:
            targets_greedy_raw = model_forward(inputs, **model_forward_kwargs)

        if tokenizer:
            targets_greedy = tokenizer.batch_decode(
                targets_greedy_raw, skip_special_tokens=True
            )
        else:
            targets_greedy = tokenizer_decode(targets_greedy_raw, **decode_kwargs)

        # generation may produce more text than just one sentence. TODO: Truncate
        # the outputs appropriately

        for output, request in zip(targets_greedy, requests):
            rq_target = request["requested_rewrite"]["target_new"]
            if not semantically_equivalent(output, rq_target):
                filtered_requests.append(request)

    return filtered_requests


def filter_requests(
    requests: List[Dict],
    model: Any = None,
    tokenizer: Any = None,
    model_forward: Optional[Callable] = None,
    tokenizer_encode: Optional[Callable] = None,
    tokenizer_decode: Optional[Callable] = None,
    model_forward_kwargs: Optional[Dict] = {},
    encode_kwargs: Optional[Dict] = {},
    decode_kwargs: Optional[Dict] = {},
) -> List[Dict]:
    """
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
        requests: List[Dict] where each Dict matches the request format specified
            in the `eme.get_request` method
        model: The target model to edit. If provided, used to get greedy completions
            for each request. Overrides `model_forward` if both are specified.
        tokenizer: The tokenizer for use with the model. Must have the same API as
            HuggingFace PreTrainedTokenizer. If using a HuggingFace model, must be
            a HuggingFace Tokenizer. Overrides `tokenizer_encode` and `tokenizer_decode`
            if all are specified.
        model_forward: A model's forward function, or a wrapper around said function
            that takes in tokenized inputs and outputs generated completions to those
            inputs. `model` must be None.
        tokenizer_encode: A tokenizer's encode function, for users who do not have
            a tokenizer matching HF's API. `tokenizer` must be None, must also
            specify `tokenizer_decode`.
        tokenizer_decode: A tokenizer's decode function, for users who do not have
            a tokenizer matching HF's API. `tokenizer` must be None, must also
            specify `tokenizer_encode`.
        encode_kwargs: Args and kwargs for use, if any, with `tokenizer_encode`
        decode_kwargs: Args and kwargs for use, if any with `tokenizer_decode`

    returns: List[Dict] consisting of a subset of `requests`
    """

    # PyTorch
    if model is None or isinstance(model, nn.Module):
        pt_filter_kwargs: Dict[str, Any] = dict()

        # build the kwargs while doing input sanity checks
        if model is None:
            assert (
                model_forward is not None
            ), "Must specify model_forward if model is None"
            pt_filter_kwargs["model_forward"] = model_forward
            pt_filter_kwargs["model_foward_kwargs"] = model_forward_kwargs
        else:
            pt_filter_kwargs["model"] = model

        if tokenizer is None:
            assert (
                tokenizer_encode is not None and tokenizer_decode is not None
            ), "Must specify both tokenizer encoding and decoding functions if `tokenizer` is None"
            pt_filter_kwargs["tokenizer_encode"] = tokenizer_encode
            pt_filter_kwargs["tokenizer_decode"] = tokenizer_decode
            pt_filter_kwargs["encode_kwargs"] = encode_kwargs
            pt_filter_kwargs["decode_kwargs"] = decode_kwargs
        else:
            pt_filter_kwargs["tokenizer"] = tokenizer

        return _filter_requests_pt(
            requests,
            **pt_filter_kwargs,
        )

    # HuggingFace
    if isinstance(model, PreTrainedModel):
        assert isinstance(
            tokenizer, PreTrainedTokenizer
        ), f"""Using a HuggingFace model without a HuggingFace tokenizer
             (got {type(tokenizer)}). Please pass in the model's associated Tokenizer."""
        return _filter_requests_hf(requests, model=model, tokenizer=tokenizer)

    raise TypeError(
        f"""Got model of type {type(model)}, only PyTorch models (nn.Module)
         and HF models (PreTrainedModel) are supported."""
    )
