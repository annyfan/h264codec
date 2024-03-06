#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import importlib
import logging
import os
from typing import Optional, Dict

import torch

# LOG with namespace identifier.
LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())

SUPPORTED_NORM_FNS = []
NORM_LAYER_REGISTRY = {}
NORM_LAYER_CLS = []


def register_norm_fn(name):
    def register_fn(cls):
        if name in SUPPORTED_NORM_FNS:
            raise ValueError(
                "Cannot register duplicate normalization function ({})".format(name)
            )
        SUPPORTED_NORM_FNS.append(name)
        NORM_LAYER_REGISTRY[name] = cls
        NORM_LAYER_CLS.append(cls)
        return cls

    return register_fn


def build_normalization_layer(
    config: Dict,
    num_features: int,
    norm_type: Optional[str] = None,
    num_groups: Optional[int] = None,
    momentum: Optional[float] = None,
) -> torch.nn.Module:
    """
    Helper function to build the normalization layer. The function can be used in either of below mentioned ways:
    Scenario 1: Set the default normalization layers using command line arguments. This is useful when the same
    normalization layer is used for the entire network (e.g., ResNet). Scenario 2: Network uses different
    normalization layers. In that case, we can override the default normalization layer by specifying the name using
    `norm_type` argument.
    """
    if norm_type is None:
        norm_type = config["model"]["normalization"]["name"]
    if num_groups is None:
        num_groups = config["model"]["normalization"]["groups"]
    if momentum is None:
        momentum = config["model"]["normalization"]["momentum"]

    norm_layer = None
    norm_type = norm_type.lower()

    if norm_type in NORM_LAYER_REGISTRY:
        device = config["model"]["device"]
        # For detecting non-cuda envs, we do not use torch.cuda.device_count() < 1
        # condition because tests always use CPU, even if cuda device is available.
        # Otherwise, we will get "ValueError: SyncBatchNorm expected input tensor to be
        # on GPU" Error when running tests on a cuda-enabled node (usually linux).
        #
        if "cuda" not in device and "sync_batch" in norm_type:
            # for a CPU-device, Sync-batch norm does not work. So, change to batch norm
            norm_type = norm_type.replace("sync_", "")
        norm_layer = NORM_LAYER_REGISTRY[norm_type](
            normalized_shape=num_features,
            num_features=num_features,
            momentum=momentum,
            num_groups=num_groups,
        )
    else:
        LOG.error(
            "Supported normalization layer arguments are: {}. Got: {}".format(
                SUPPORTED_NORM_FNS, norm_type
            )
        )
    return norm_layer


# automatically import different normalization layers
norm_dir = os.path.dirname(__file__)
for file in os.listdir(norm_dir):
    path = os.path.join(norm_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        model_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module(
            "h264.src.models.modules.byteformer.layers.normalization." + model_name
        )
