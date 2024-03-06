#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import logging
import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import init

from h264.src.models.modules.byteformer.layers import (
    positional_embedding,
    token_merging,
)
from h264.src.models.modules.byteformer.windowed_transformer import (
    WindowedTransformerEncoder,
)
from h264.src.models.modules.byteformer.layers import embedding
from h264.src.models.modules.byteformer.layers.normalization import (
    build_normalization_layer,
)
from h264.src.models.utils import unfold_tokens

# from h264.src.utils.common import get_value_from_config

# LOG with namespace identifier.
LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


class ByteFormer(nn.Module):
    """
    This class defines the `ByteFormer <https://arxiv.org/pdf/2306.00238.pdf>`_
    architecture.
    """

    def __init__(self, config: Dict) -> None:
        super(ByteFormer, self).__init__()

        self.config = config
        self.device = config["model"]["device"]

        embed_dim = config["model"]["byteformer"]["embed_dim"]
        ffn_dim = config["model"]["byteformer"]["ffn_dim"]
        n_transformer_layers = config["model"]["byteformer"]["n_transformer_layers"]
        num_heads = config["model"]["byteformer"]["n_attn_heads"]
        attn_dropout = config["model"]["byteformer"]["attn_dropout"]
        dropout = config["model"]["byteformer"]["dropout"]
        ffn_dropout = config["model"]["byteformer"]["ffn_dropout"]
        norm_layer = config["model"]["byteformer"]["norm_layer"]

        # The vocab size of the token embedding. Defaults to 257,corresponding
        # to the number of unique bytes (256) plus 1 more for the mask token.
        vocab_size = config["model"]["byteformer"]["vocab_size"]
        self.embeddings = embedding.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=-1
        )
        # Reinitialize everything except the padding index.
        init.trunc_normal_(self.embeddings.weight[:-1], std=math.sqrt(1.0 / embed_dim))

        # The token length to use for dummy inputs. Defaults to 48564, corresponding
        # to the average length of 224x224 JPEG images from ImageNet.
        self.dummy_input_token_length = config["model"]["byteformer"][
            "dummy_input_token_length"
        ]

        # Add token reduction convolution.
        # The size of the kernel of the initial downsampling conv1d. Defaults to 16.
        self.conv_kernel_size = config["model"]["byteformer"]["conv_kernel_size"]
        self.conv_stride = config["model"]["byteformer"]["conv_stride"]
        #self.deconv_kernel_size = config["model"]["byteformer"]["deconv_kernel_size"]
        #self.deconv_stride = config["model"]["byteformer"]["deconv_stride"]

        if self.conv_kernel_size == 0:
            # We skip the convolution.
            self.token_reduction_net = None
        if self.conv_kernel_size is not None:
            #self.deconv = nn.ConvTranspose1d(
            #    embed_dim,
            #    embed_dim,
            #    kernel_size=self.deconv_kernel_size,
            #    stride=self.deconv_kernel_size,
            #    padding = 0,
            #    bias=False,
            #)
            self.token_reduction_net = nn.Conv1d(
                embed_dim,
                embed_dim,
                kernel_size=self.conv_kernel_size,
                stride=self.conv_stride,
                bias=False,
            )

        # Add the positional embeddings.
        # The maximum number of tokens that can be input to the network. Defaults to 50000.
        self.max_num_tokens = config["model"]["byteformer"]["max_num_tokens"]

        # Use sinusoidal instead of learnable positional encoding. Defaults to False.
        self.sinusoidal_pos_embed = config["model"]["byteformer"]["sinusoidal_pos_emb"]
        self.pos_embed = positional_embedding.PositionalEmbedding(
            num_embeddings=self.max_num_tokens,
            embedding_dim=embed_dim,
            sequence_first=False,
            padding_idx=None,
            is_learnable=not self.sinusoidal_pos_embed,
            interpolation_mode="bilinear",
        )

        # Dropout in Byteformer layers. Defaults to 0.0.
        pos_emb_drop_p = config["model"]["byteformer"]["pos_emb_drop_p"]
        self.emb_dropout = nn.Dropout(p=pos_emb_drop_p)

        # Build the transformer backbone.
        # A list of window sizes used in shifted window attention.
        # If the list is length 1, the same window size is used for all windows.
        # Defaults to 128 for all windows.
        window_sizes = config["model"]["byteformer"]["window_sizes"]

        # A list of shifts used in shifted window attention. Defaults to values
        # that alternate between 0 and 64.
        window_shifts = [0, 64] * 6

        # A list of boolean values, where the i'th element specifies whether to
        # downsample after the transformer block with index i.  Defaults to default_downsampling.
        downsample = [True, True] + ([False, True] * 4) + [False, False]

        if len(window_sizes) == 1:
            window_sizes = window_sizes * n_transformer_layers

        for x in [window_sizes, window_shifts, downsample]:
            if len(x) != n_transformer_layers:
                raise ValueError(
                    f"Invalid argument length {len(x)} != {n_transformer_layers}"
                )
        # Probability of applying stochastic dropout to TransformerEncoder submodules.
        # Defaults to 0.0
        stochastic_dropout = config["model"]["byteformer"]["stochastic_dropout"]
        per_layer_stochastic_drop_rate = [
            round(x, 3)
            for x in np.linspace(0, stochastic_dropout, n_transformer_layers)
        ]

        blocks = []
        self.downsamplers = nn.ModuleDict()
        for layer_idx in range(n_transformer_layers):
            blocks.append(
                WindowedTransformerEncoder(
                    config=config,
                    embed_dim=embed_dim,
                    ffn_latent_dim=ffn_dim,
                    num_heads=num_heads,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    ffn_dropout=ffn_dropout,
                    transformer_norm_layer=norm_layer,
                    stochastic_dropout=per_layer_stochastic_drop_rate[layer_idx],
                    window_size=window_sizes[layer_idx],
                    window_shift=window_shifts[layer_idx],
                )
            )
            if downsample is not None and downsample[layer_idx]:
                self.downsamplers[
                    self.get_downsampler_name(layer_idx)
                ] = token_merging.TokenMerging(embed_dim)
        self.transformer = nn.Sequential(*blocks)

        self.post_transformer_norm = build_normalization_layer(
            config, num_features=embed_dim, norm_type=norm_layer
        )

        # num_classes = getattr(opts, "model.regression.n_classes")
        # self.classifier = LinearLayer(embed_dim, num_classes)

    def dummy_input_and_label(self, batch_size: int) -> Dict:
        """
        Get a dummy input and label that could be passed to the model.

        Args:
            batch_size: The batch size to use for the generated inputs.

        Returns:
            A dict with
                {
                    "samples": tensor of shape [batch_size, sequence_length],
                    "targets": tensor of shape [batch_size],
                }
        """
        n_labels = 10
        max_value = 257

        samples = torch.randint(
            0, max_value, [batch_size, self.dummy_input_token_length]
        )
        images = torch.randint(0, max_value, [batch_size, 3, 244, 244])
        return {"h264": samples, "image": images}

    def apply_token_reduction_net(
        self, x: Tensor, x_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply the portion of the network used to reduce sequence lengths before
        the transformer backbone.

        Args:
            x: The input token embeddings of shape [batch_size, sequence_length,
                embed_dim].
            x_mask: The input mask of shape [batch_size, sequence_length].

        Returns:
            New versions of @x and @x_mask, downsampled along the sequence
            dimension by the token reduction net.
        """
        B, N, C = x.shape
        if self.token_reduction_net is None:
            return x, x_mask

        #x = self.deconv(x.permute(0, 2, 1)).permute(0, 2, 1)
        #if x_mask is not None:
        #    #(L in−1)×stride−2×padding+dilation×(kernel_size−1)+output_padding+1
        #    x_mask = x_mask.reshape(B, N, 1).float().repeat_interleave(self.deconv_kernel_size,dim=2)
        #    x_mask = x_mask.view(x.shape[0], x.shape[1])

        #N2 = x.shape[1]
        x = self.token_reduction_net(x.permute(0, 2, 1)).permute(0, 2, 1)
        if x_mask is not None:
            x_mask = unfold_tokens(
            #    x_mask.reshape(B, N2, 1).float(), self.conv_kernel_size, self.conv_stride
                 x_mask.reshape(B, N, 1).float(), self.conv_kernel_size, self.conv_stride
            )
            # The mask is now [B * N, kernel_size, 1]. It contains values in {0, -inf}.
            x_mask = x_mask.max(dim=1).values.view(x.shape[0], x.shape[1])

            # assert x.shape[:2] == x_mask.shape
        return x, x_mask

    def get_backbone_inputs(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Convert input bytes into embeddings to be passed to the network's
        transformer backbone.

        Args:
            x: The input bytes as an integer tensor of shape [batch_size,
                sequence_length]. Integer tensors are expected (rather than byte
                tensors) since -1 is usually used for padding.

        Returns:
            The embeddings of shape [batch_size, new_sequence_length] and a
            mask tensor of shape [batch_size, new_sequence_length]. The mask
            contains 0 at unmasked positions and float(-inf) at masked
            positions.
        """
        mask = torch.zeros_like(x, dtype=torch.float)
        mask[x == -1] = mask[x == -1].fill_(float("-inf"))
        #mask[x == -1].fill_(float("-inf"))
        mask = mask.detach().requires_grad_(False)
        x[x == -1] = self.embeddings.padding_idx
        x = self.embeddings(x)

        x, mask = self.apply_token_reduction_net(x, mask)
        x = x + self.pos_embed(self.max_num_tokens)[:, : x.shape[1]]

        x = self.emb_dropout(x)
        return x, mask

    def backbone_forward(
        self, x: Tensor, key_padding_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Execute the forward pass of the network's transformer backbone.

        Args:
            x: The input embeddings as a [batch_size, sequence_length, embed_dim] tensor.
            key_padding_mask: The mask tensor of shape [batch_size, sequence_length].

        Returns:
            The outputs of the backbone as a tuple. The first element is the feature
            tensor, and the second element is the updated key_padding_mask.
        """
        B, S, _ = x.shape
        # assert key_padding_mask.shape == (B, S)

        for layer_idx, elem in enumerate(self.transformer):
            x = elem(x, key_padding_mask=key_padding_mask)
            if self.get_downsampler(layer_idx) is not None:
                x, key_padding_mask = self.get_downsampler(layer_idx)(
                    x, key_padding_mask
                )
        x = self.post_transformer_norm(x)
        return x, key_padding_mask

    def get_downsampler_name(self, idx: int) -> str:
        """
        Get the name of the downsampling layer with index @idx.

        Args:
            idx: The index of the downsampling layer.

        Returns:
            A string representing the name of the donwsampling layer.
        """
        return f"downsample_{idx}"

    def get_downsampler(self, idx: int) -> Optional[nn.Module]:
        """
        Get the module that performs downsampling after transformer layer @idx.
        If no downsampling occurs after that layer, return None.

        Args:
            idx: The desired index.

        Returns:
            The downsampling layer, or None.
        """
        name = self.get_downsampler_name(idx)
        if name not in self.downsamplers:
            return None
        return self.downsamplers[name]

    def forward(self, x: Tensor, *args, **kwargs) -> Dict:
        """
        Perform a forward pass on input bytes. The tensor is
        stored as an integer tensor of shape [batch_size, sequence_length].
        Integer tensors are used because @x usually contains mask tokens.

        Args:
            x: The input tensor of shape [batch_size, sequence_length].

        Returns:
            The output logits.
        """

        x, key_padding_mask = self.get_backbone_inputs(x)
        x, attn_mask = self.backbone_forward(x, key_padding_mask)
        # b * sequence_length * embeding size
        attn_mask_reshape = attn_mask.view(x.shape[0], x.shape[1], 1)
        x[(attn_mask_reshape == float("-inf")).expand(-1, -1, x.shape[-1])] = 0

        last_hidden_state = x

        # b * embeding size, mean pool across the word dimension
        norms = (attn_mask_reshape == 0).sum(dim=1)
        x = torch.sum(x, dim=1) / norms

        # x = self.classifier(x)

        return {"c": x, "last_hidden_state": last_hidden_state, "attn_mask": attn_mask}
