# ----------------------------------------------------------------------------------------------
# Copyright (C) B  - All Rights Reserved
# Unauthorized copying, use, or modification to this file via any medium is strictly prohibited.
# This file is private and confidential.
# Contact: dev@b
# ----------------------------------------------------------------------------------------------

import hashlib
import os
import requests
import logging
from functools import wraps
from math import ceil
from typing import Any, List, Union, Dict, Tuple, Iterable

import torch
from torch import Tensor
from torch.utils.data.sampler import Sampler
import tqdm

# Define the logger for this library.
LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


def get_tensor_sizes(data: Union[Dict, Tensor]) -> Union[List[str], List[Tuple[int]]]:
    """Utility function for extracting tensor shapes (for printing purposes only)."""
    if isinstance(data, Dict):
        tensor_sizes = []
        for k, v in data.items():
            size_ = get_tensor_sizes(v)
            if size_:
                tensor_sizes.append(f"{k}: {size_}")
        return tensor_sizes
    elif isinstance(data, Tensor):
        return [*data.shape]
    else:
        return []


def unfold_tokens(t: Tensor, kernel_size: int, step:int) -> Tensor:
    """
    Group tokens from tensor @t using torch.Tensor.unfold, using the given
    kernel size. This amounts to windowing @t using overlapping windows
    of size @kernel_size, with overlap of @kernel_size // 2.

    Args:
        t: A tensor of shape [batch_size, sequence_length, num_channels].
        kernel_size: The kernel size.

    Returns:
        A tensor of shape [batch_size * (sequence_length - kernel_size)
        // (kernel_size // 2) + 1, kernel_size, num_channels].
    """
    t = t.unfold(dimension=1, size=kernel_size, step=step)
    B, L, C, _ = t.shape
    t = t.reshape(B * L, C, kernel_size)
    t = t.transpose(1, 2)
    return t


def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def find_first(fn, arr):
    for ind, el in enumerate(arr):
        if fn(el):
            return el
    return -1

def split_iterable(it, split_size):
    accum = []
    for ind in range(ceil(len(it) / split_size)):
        start_index = ind * split_size
        accum.append(it[start_index : (start_index + split_size)])
    return accum

def split(t, split_size=None):
    if not exists(split_size):
        return t

    if isinstance(t, torch.Tensor):
        return t.split(split_size, dim=0)

    if isinstance(t, Iterable):
        return split_iterable(t, split_size)

    return TypeError


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def split_batch(split_size=None, batch=None):
    data = batch
    if split_size is None:
        yield 1, batch

    first_tensor = find_first(lambda t: isinstance(t, torch.Tensor), [*batch.values()])
    assert exists(first_tensor)

    batch_size = len(first_tensor)
    split_size = default(split_size, batch_size)
    dict_keys = data.keys()
    num_chunks = ceil(batch_size / split_size)
    split_all_batches = [
        split(arg, split_size=split_size)
        if exists(arg) and isinstance(arg, (torch.Tensor, Iterable))
        else ((arg,) * num_chunks)
        for arg in batch.values()
    ]
    chunk_sizes = num_to_groups(batch_size, split_size)

    for chunk_size, *chunked_batch_values in tuple(
        zip(chunk_sizes, *split_all_batches)
    ):
        chunked_batch = dict(tuple(zip(dict_keys, chunked_batch_values)))
        chunk_size_frac = chunk_size / batch_size
        yield chunk_size_frac, chunked_batch


# def imagen_sample_in_chunks(fn):
#    @wraps(fn)
#    def inner(self, *args, max_batch_size=None, **kwargs):
#        if not exists(max_batch_size):
#            return fn(self, *args, **kwargs)
#
#        if self.imagen.unconditional:
#            batch_size = kwargs.get("batch_size")
#            batch_sizes = num_to_groups(batch_size, max_batch_size)
#            outputs = [
#                fn(self, *args, **{**kwargs, "batch_size": sub_batch_size})
#                for sub_batch_size in batch_sizes
#            ]
#        else:
#            outputs = [
#                fn(self, *chunked_args, **chunked_kwargs)
#                for _, (chunked_args, chunked_kwargs) in split_batch(
#                    split_size=max_batch_size, **kwargs
#                )
#            ]
#
#        if isinstance(outputs[0], torch.Tensor):
#            return torch.cat(outputs, dim=0)
#
#        return list(map(lambda t: torch.cat(t, dim=0), list(zip(*outputs))))
#
#    return inner
#


# image normalization functions
# images to be in the range of -1 to 1
def normalize_neg_one_to_one(img):
    """
    img: image 0 to 1
    return image -1 to 1
    """
    return img * 2 - 1


def unnormalize_zero_to_one(normed_img):
    """
    img: image -1 to 1
    return image 0 to 1
    """
    return (normed_img + 1) * 0.5


def restore_parts(state_dict_target, state_dict_from):
    for name, param in state_dict_from.items():

        if name not in state_dict_target:
            continue

        if param.size() == state_dict_target[name].size():
            state_dict_target[name].copy_(param)
        else:
            print(f"layer {name}({param.size()} different than target: {state_dict_target[name].size()}")

    return state_dict_target