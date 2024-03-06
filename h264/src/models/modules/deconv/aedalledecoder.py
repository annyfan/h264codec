from typing import Dict
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from h264.src.models.modules.byteformer.layers.normalization.layer_norm import LayerNorm2D_NCHW
from h264.src.models.modules.deconv.dalledecoder2 import Decoder

from h264.src.models.utils import normalize_neg_one_to_one, unfold_tokens, unnormalize_zero_to_one

class DeconvDecoder(nn.Module):
    """Vanilla Variational Autoencoder decoder.

    Args:
        model_config (config): The Variational Autoencoder configuration setting the main
        parameters of the model.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(
        self,
        model_config: Dict
    ):
        super().__init__()

        self.model_name = "AE"
        embeding_dim = model_config["model"]["decovdecoder"]["embeding_dim"]
        self.embeding_dim = embeding_dim

        
        self.decoder = Decoder(vocab_size=embeding_dim,requires_grad= True, n_init=model_config["model"]["decovdecoder"]["n_init"])


        
    

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x 2 x 2]
        :return: (Tensor) [B x C x H x W]
        """
   
        result = self.decoder(z)
        return result

    def forward(self, x: Tensor, **kwargs):
        """
        The VAE model

        Args:
            x : The training input

        Returns:
            recon_loss, loss, recon_x

        """
        # Split the result into mu and var components
        # of the latent Gaussian distribution
      
        recon_x = self.decode(x)

        return recon_x


class ReduceEmbeding(nn.Module):
    """
    input: [B xLx D]
    return: (Tensor) [B x hidden_dims x 4]
    """

    def __init__(
        self,
        hidden_dims,
        dropout = 0.0
        
    ):
        super().__init__()

        modules = []
        self.output_dim = hidden_dims[-1]
        for i in range(len(hidden_dims) - 1):
            modules.append( 
                nn.Sequential(
                    nn.Conv1d(hidden_dims[i],
                              hidden_dims[i + 1],
                              kernel_size=1,
                              stride = 1),
                    nn.LeakyReLU(),
                    nn.Conv1d(hidden_dims[i + 1],
                              hidden_dims[i + 1],
                              kernel_size=1,
                              stride = 1),
                    LayerNorm2D_NCHW(hidden_dims[i + 1]),
                    nn.LeakyReLU()),
                    
                )
        
        self.conv_layers = nn.Sequential(*modules)   
        self.global_pooling = nn.AdaptiveMaxPool1d(1)
        self.decoder_input = nn.Sequential(
                    nn.Linear(self.output_dim, self.output_dim * 2),
                    nn.LeakyReLU(),
                    nn.Linear(self.output_dim * 2, self.output_dim * 4),
                    nn.LeakyReLU(),
                    nn.Dropout(dropout))


    def forward(self, x: Tensor, x_mask : Tensor, **kwargs):
        """
        The VAE model

        param z: (Tensor) [B L D]
        return: (Tensor) [B E 2 2 ]

        """
        B, N, C = x.shape # (batch_size,  seq_length, embedding_size)
        # Transpose tensor to fit Conv1d input shape
        # Apply Conv1d layers to adjust embedding size and sequence length
        adjusted_x = self.conv_layers(x.permute(0, 2, 1)).permute(0, 2, 1)  

        if x_mask is not None:
            attn_mask_reshape = x_mask.view(B, N, 1)
            #adjusted_x[(attn_mask_reshape == float("-inf")).expand(-1, -1, adjusted_x.shape[-1])] = 0
            adjusted_x[(attn_mask_reshape == float("-inf")).expand(-1, -1, adjusted_x.shape[-1])] = float("-inf")

            # (batch_size,  output_embedding_size), mean pool across the word dimension
            #norms = (attn_mask_reshape == 0).sum(dim=1)
            #x_pooled = torch.sum(adjusted_x, dim=1) / norms
            x_pooled = self.global_pooling(adjusted_x.permute(0, 2, 1)).squeeze(-1)
        else:
            # Perform global average pooling or global max pooling to remove sequence length dimension
            x_pooled = self.global_pooling(adjusted_x.permute(0, 2, 1)).squeeze(-1)  # (batch_size,  output_embedding_size)

        x_output = self.decoder_input(x_pooled)
        x_output =  x_output.view(-1, self.output_dim, 2, 2)
        
        return x_output
