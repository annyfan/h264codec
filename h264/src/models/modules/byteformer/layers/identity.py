#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

from torch import Tensor, nn


class Identity(nn.Module):
    """
    This is a place-holder and returns the same tensor.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        return x
