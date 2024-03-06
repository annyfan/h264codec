#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
import argparse
import logging
from typing import Optional, Dict

from torch import Tensor, nn
from torch.nn import Dropout

from h264.src.models.modules.byteformer.layers.activation import build_activation_layer

from h264.src.models.modules.byteformer.layers.identity import Identity
from h264.src.models.modules.byteformer.layers.linear_layer import LinearLayer
from h264.src.models.modules.byteformer.layers.multi_head_attention import (
    MultiHeadAttention,
)
from h264.src.models.modules.byteformer.layers.normalization import (
    build_normalization_layer,
)
from h264.src.models.modules.byteformer.layers.single_head_attention import (
    SingleHeadAttention,
)
from torchvision.ops import StochasticDepth as StochasticDepthTorch

# LOG with namespace identifier.
LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


class TransformerEncoder(nn.Module):
    """
    This class defines the pre-norm `Transformer encoder <https://arxiv.org/abs/1706.03762>`_
    Args:
        opts: Command line arguments.
        embed_dim: :math:`C_{in}` from an expected input of size :math:`(N, P, C_{in})`.
        ffn_latent_dim: Inner dimension of the FFN.
        num_heads: Number of heads in multi-head attention. Default: 8.
        attn_dropout: Dropout rate for attention in multi-head attention. Default: 0.0
        dropout: Dropout rate. Default: 0.0.
        ffn_dropout: Dropout between FFN layers. Default: 0.0.
        transformer_norm_layer: Normalization layer. Default: layer_norm.
        stochastic_dropout: Stochastic dropout setting. Default: 0.0.

    Shape:
        - Input: :math:`(N, P, C_{in})` where :math:`N` is batch size, :math:`P` is number of patches,
        and :math:`C_{in}` is input embedding dim
        - Output: same shape as the input
    """

    def __init__(
        self,
        config: Dict,
        embed_dim: int,
        ffn_latent_dim: int,
        num_heads: Optional[int] = 8,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.0,
        ffn_dropout: Optional[float] = 0.0,
        transformer_norm_layer: Optional[str] = "layer_norm",
        stochastic_dropout: Optional[float] = 0.0,
        enable_coreml_compatible_module: Optional[bool] = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        attn_unit = SingleHeadAttention(
            embed_dim=embed_dim, attn_dropout=attn_dropout, bias=True
        )
        if num_heads > 1:
            attn_unit = MultiHeadAttention(
                embed_dim,
                num_heads,
                attn_dropout=attn_dropout,
                bias=True,
                coreml_compatible=enable_coreml_compatible_module,
            )

        self.pre_norm_mha = nn.Sequential(
            build_normalization_layer(
                config, norm_type=transformer_norm_layer, num_features=embed_dim
            ),
            attn_unit,
            Dropout(p=dropout),
        )

        act_name = build_activation_layer(config=config, num_parameters=1)
        self.pre_norm_ffn = nn.Sequential(
            build_normalization_layer(
                config, norm_type=transformer_norm_layer, num_features=embed_dim
            ),
            LinearLayer(in_features=embed_dim, out_features=ffn_latent_dim, bias=True),
            act_name,
            Dropout(p=ffn_dropout),
            LinearLayer(in_features=ffn_latent_dim, out_features=embed_dim, bias=True),
            Dropout(p=dropout),
        )

        self.drop_path = Identity()
        if stochastic_dropout > 0.0:
            if dropout > 0.0:
                LOG.error(
                    "Stochastic dropout and dropout are mutually exclusive. "
                    "Use either of them, but not both."
                    "Got: {} and {}".format(stochastic_dropout, dropout)
                )
            self.drop_path = StochasticDepthTorch(p=stochastic_dropout, mode="row")

        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.ffn_dropout = ffn_dropout
        self.stochastic_dropout = stochastic_dropout
        self.std_dropout = dropout
        self.attn_fn_name = attn_unit.__class__.__name__
        self.act_fn_name = act_name.__class__.__name__
        self.norm_type = transformer_norm_layer

    def __repr__(self) -> str:
        return "{}(embed_dim={}, ffn_dim={}, dropout={}, ffn_dropout={}, stochastic_dropout={}, attn_fn={}, act_fn={}, norm_fn={})".format(
            self.__class__.__name__,
            self.embed_dim,
            self.ffn_dim,
            self.std_dropout,
            self.ffn_dropout,
            self.stochastic_dropout,
            self.attn_fn_name,
            self.act_fn_name,
            self.norm_type,
        )

    def forward(
        self,
        x: Tensor,
        x_prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        *args,
        **kwargs,
    ) -> Tensor:
        # Multi-head attention
        res = x
        x = self.pre_norm_mha[0](x)  # norm
        x = self.pre_norm_mha[1](
            x_q=x,
            x_kv=x_prev,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            *args,
            **kwargs,
        )  # mha

        x = self.drop_path(self.pre_norm_mha[2](x))  # applying stochastic depth
        x = x + res

        # Feed forward network
        x = x + self.drop_path(self.pre_norm_ffn(x))
        return x
