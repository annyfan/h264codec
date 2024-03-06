#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import argparse
import logging
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

# LOG with namespace identifier.
LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


class LinearLayer(nn.Module):
    """
    Applies a linear transformation to the input data

    Args:
        in_features (int): number of features in the input tensor
        out_features (int): number of features in the output tensor
        bias  (Optional[bool]): use bias or not
        channel_first (Optional[bool]): Channels are first or last dimension. If first, then use Conv2d

    Shape:
        - Input: :math:`(N, *, C_{in})` if not channel_first else :math:`(N, C_{in}, *)` where :math:`*` means any number of dimensions.
        - Output: :math:`(N, *, C_{out})` if not channel_first else :math:`(N, C_{out}, *)`

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: Optional[bool] = True,
        channel_first: Optional[bool] = False,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None

        self.in_features = in_features
        self.out_features = out_features
        self.channel_first = channel_first

        self.reset_params()

    def reset_params(self):
        if self.weight is not None:
            torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        if self.channel_first:
            if not self.training:
                LOG.error("Channel-first mode is only supported during inference")
            if x.dim() != 4:
                LOG.error("Input should be 4D, i.e., (B, C, H, W) format")
            # only run during conversion
            with torch.no_grad():
                return F.conv2d(
                    input=x,
                    weight=self.weight.clone()
                    .detach()
                    .reshape(self.out_features, self.in_features, 1, 1),
                    bias=self.bias,
                )
        else:
            x = F.linear(x, weight=self.weight, bias=self.bias)
        return x

    def __repr__(self):
        repr_str = (
            "{}(in_features={}, out_features={}, bias={}, channel_first={})".format(
                self.__class__.__name__,
                self.in_features,
                self.out_features,
                True if self.bias is not None else False,
                self.channel_first,
            )
        )
        return repr_str
