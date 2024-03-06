#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

from torch import nn
from . import register_act_fn


@register_act_fn(name="tanh")
class Tanh(nn.Tanh):
    """
    Applies Tanh function
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
