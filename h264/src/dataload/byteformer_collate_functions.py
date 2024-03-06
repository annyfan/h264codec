"""This file contains collate functions used by ByteFormer.

Since the model operates on a variety of input types, these collate functions
are not associated with a particular dataset.

These transforms are applied before the model (rather than inside the model) to
take advantage of parallelism, and to avoid the need to move tensors from the
GPU, back to the CPU, then back to GPU (since these transforms cannot be done
on GPU).
"""

import argparse
from typing import Dict, List, Mapping, Optional, Union, Tuple

import torch
from torch import Tensor
from torch.nn import functional
from torch.utils.data import default_collate


class H264Save(object):
    """
    Encode audio with a supported file encoding.

    Args:
        opts: The global options.
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self, data: Dict[str, Union[Dict[str, Tensor], Tensor, int]]
    ) -> Dict[str, Union[Dict[str, Tensor], Tensor, int]]:
        """
        Serialize the input as file bytes.

        Args:
            data:  (image, h264)

        Returns:
            The transformed data.
        """
        h264 = data["h264"]
        # Convert from uint8 to int32 so we can use negative values as padding.
        # The copy operation is required to avoid a warning about non-writable
        # tensors.
        # buf = h264.to(dtype=torch.int32)
        data["h264"] = h264.to(dtype=torch.int32)

        # data["h264"] = buf

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def apply_h264_save(batch: List[Mapping[str, Tensor]]) -> List[Mapping[str, Tensor]]:
    """
    Apply the TorchaudioSave transform to each batch element.

    Args:
        batch: The batch of data.

    Returns:
        The modified batch.
    """

    transform = H264Save()
    for i, elem in enumerate(batch):
        batch[i] = transform(elem)
    return batch


def apply_h264_padding(
    batch: List[Mapping[str, Tensor]],
    padding_index,
    max_seq_length=None,
    key: Optional[str] = None,
) -> List[Mapping[str, Tensor]]:
    """
    Apply padding to make samples the same length.

    The input is a list of dictionaries of the form:
        [{"samples": @entry, ...}, ...].
    If @key is specified, @entry has the form {@key: @value}, where @value
    corresponds to the entry that should be padded. Otherwise, @entry is assumed
    to be a tensor.

    The tensor mentioned in the above paragraph will have shape [batch_size,
        sequence_length, ...].

    Args:
        batch: The batch of data.
        padding_index: fill value for padding
        max_seq_length: the max seq length allowed in the model
        key: The key of the sample element to pad. If @key is None, the entry
            is assumed to be a tensor.

    Returns:
        The modified batch of size [batch_size, padded_sequence_length, ...].
    """

    if batch[0]["h264"].dim() != 1:
        # Padding only applies to 1d tensors.
        return batch
    # Tensors have shape [batch_size, sequence_length, ...]. Get the maximum
    # sequence length.
    if max_seq_length is None:
        padded_seq_len = max(be["h264"].shape[0] for be in batch)
    else:
        padded_seq_len = max_seq_length

    for elem in batch:
        sample = elem["h264"]  # [batch_size, sequence_length, ...].
        if sample.shape[0] < padded_seq_len:
            sample = functional.pad(
                sample, (0, padded_seq_len - sample.shape[0]), value=padding_index
            )  # [batch_size, padded_sequence_length, ...].
        else:
            sample = sample[0:padded_seq_len]

        if isinstance(elem["h264"], dict):
            elem["h264"][key] = sample
        else:
            elem["h264"] = sample
    return batch


def byteformer_h264_collate_fn(
    batch: List[Mapping[str, Tensor]], config: Dict
) -> List[Mapping[str, Tensor]]:
    """
    Apply augmentations specific to ByteFormer audio training, then perform
    padded collation.

    See `ByteFormer <https://arxiv.org/abs/2212.10553>`_ for more information
    on this modeling approach.

    Args:
        config: model config Dict
        batch: The batch of data.

    Returns:
        The modified batch.
    """

    padding_index = config["model"]["byteformer"]["padding_index"]
    max_seq_length = config["dataset"].get("h264_padded_seq_len", None)
    batch = apply_h264_save(batch)
    batch = apply_h264_padding(
        batch, max_seq_length=max_seq_length, padding_index=padding_index
    )
    batch = default_collate(batch)
    return batch
