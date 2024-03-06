
import argparse
import os
import random
import re
from types import MethodType
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np


import torch
from torch import Tensor

import logging

# Set the logging for this application.
LOG = logging.getLogger(os.path.basename(__file__))

def get_value_from_config(configs: Dict, name: str, default: Any) -> Any:
    """
    get value from config, if None, then return default

    Args:
        configs: config Dictionary
        name: config name
        default: default value

    Returns:
        the config value
    """
    if name is None or configs is None:
        return default
    keys = name.split('.')
    if keys is None or len(keys) == 0:
        return default
    config_object = configs
    try:
        for key in keys:
            config_object = config_object[key]
        return config_object
    except (KeyError, TypeError):
        return default
    
def clean_strip(
    obj: Union[str, List[str]], sep: Optional[str] = ",", strip: bool = True
) -> List[str]:
    # Allowing list of strings as input as well as comma-separated strings
    if isinstance(obj, list):
        strings = obj
    else:
        strings = obj.split(sep)

    if strip:
        strings = [x.strip() for x in strings]
    strings = [x for x in strings if x]
    return strings


def load_pretrained_model(
    model: torch.nn.Module, wt_loc: str, config: Dict, *args, **kwargs
) -> torch.nn.Module:
    """Helper function to load pre-trained weights.
    Args:
        model: Model whose weights will be loaded.
        wt_loc: Path to file to load state_dict from.
        opts: Input arguments.
    Returns:
        The model loaded with the given weights.

    """
    if not os.path.isfile(wt_loc):
        LOG.error("Pretrained file is not found here: {}".format(wt_loc))

    wts = torch.load(wt_loc, map_location="cpu")

    exclude_scopes = get_value_from_config(config, "model.resume_exclude_scopes", "")
    exclude_scopes: List[str] = clean_strip(exclude_scopes)

    missing_scopes = get_value_from_config(config,  "model.ignore_missing_scopes", "")
    missing_scopes: List[str] = clean_strip(missing_scopes)

    rename_scopes_map: List[List[str]] =get_value_from_config(config, "model.rename_scopes_map", [])
    if rename_scopes_map:
        for entry in rename_scopes_map:
            if len(entry) != 2:
                raise ValueError(
                    "Every entry in model.rename_scopes_map must contain exactly two string elements"
                    " for before and after. Got {}.".format(str(entry))
                )

    # By default, adding scopes that we exclude to missing scopes
    # If you excluded something, you can't expect it to be there.
    missing_scopes += exclude_scopes

    # remove unwanted scopes
    if exclude_scopes:
        for key in wts.copy():
            if any([re.match(x, key) for x in exclude_scopes]):
                del wts[key]

    if rename_scopes_map:
        for before, after in rename_scopes_map:
            wts = {re.sub(before, after, key): value for key, value in wts.items()}

    strict = not bool(missing_scopes)

    try:
        module = unwrap_model_fn(model)
        missing_keys, unexpected_keys = module.load_state_dict(wts, strict=strict)

        if unexpected_keys:
            raise Exception(
                "Found unexpected keys: {}."
                "You can ignore these keys using `model.resume_exclude_scopes`.".format(
                    ",".join(unexpected_keys)
                )
            )

        missing_keys = [
            key
            for key in missing_keys
            if not any([re.match(x, key) for x in missing_scopes])
        ]

        if missing_keys:
            raise Exception(
                "Missing keys detected. Did not find the following keys in pre-trained model: {}."
                " You can ignore the keys using `model.ignore_missing_scopes`.".format(
                    ",".join(missing_keys)
                )
            )

            LOG.info("Pretrained weights are loaded from {}".format(wt_loc))
    except Exception as e:
    
            LOG.error(
                "Unable to load pretrained weights from {}. Error: {}".format(wt_loc, e)
            )

    return model


def parameter_list(
    named_parameters,
    weight_decay: Optional[float] = 0.0,
    no_decay_bn_filter_bias: Optional[bool] = False,
    *args,
    **kwargs,
) -> List[Dict]:
    module_name = kwargs.get("module_name", "")
    with_decay = []
    without_decay = []
    with_decay_param_names = []
    without_decay_param_names = []
    if isinstance(named_parameters, list):
        for n_parameter in named_parameters:
            for p_name, param in n_parameter():
                if (
                    param.requires_grad
                    and len(param.shape) == 1
                    and no_decay_bn_filter_bias
                ):
                    # biases and normalization layer parameters are of len 1
                    without_decay.append(param)
                    without_decay_param_names.append(module_name + p_name)
                elif param.requires_grad:
                    with_decay.append(param)
                    with_decay_param_names.append(module_name + p_name)
    else:
        for p_name, param in named_parameters():
            if (
                param.requires_grad
                and len(param.shape) == 1
                and no_decay_bn_filter_bias
            ):
                # biases and normalization layer parameters are of len 1
                without_decay.append(param)
                without_decay_param_names.append(module_name + p_name)
            elif param.requires_grad:
                with_decay.append(param)
                with_decay_param_names.append(module_name + p_name)
    param_list = [
        {
            "params": with_decay,
            "weight_decay": weight_decay,
            "param_names": with_decay_param_names,
        }
    ]
    if len(without_decay) > 0:
        param_list.append(
            {
                "params": without_decay,
                "weight_decay": 0.0,
                "param_names": without_decay_param_names,
            }
        )
    return param_list


def freeze_module(module: torch.nn.Module, force_eval: bool = True) -> torch.nn.Module:
    """
    Sets requires_grad = False on all the given module parameters, and put the module in eval mode.
    By default, it also overrides the module's `train` method to make sure that it always stays in eval mode
    (ie calling ``module.train(mode=True)`` executes ``module.train(mode=False)``)

    >>> module = nn.Linear(10, 20).train()
    >>> module.training
    True
    >>> module.weight.requires_grad
    True
    >>> freeze_module(module).train().training
    False
    >>> module.weight.requires_grad
    False
    """

    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False

    if force_eval:

        def _force_train_in_eval(
            self: torch.nn.Module, mode: bool = True
        ) -> torch.nn.Module:
            # ignore train/eval calls: perpetually stays in eval
            return self

        module.train = MethodType(_force_train_in_eval, module)

    return module


def freeze_modules_based_on_config(
    config: Dict, model: torch.nn.Module, verbose: bool = True
) -> torch.nn.Module:
    """
    Allows for freezing immediate modules and parameters of the model using --model.freeze-modules.

    --model.freeze-modules should be a list of strings or a comma-separated list of regex expressions.

    Examples of --model.freeze-modules:
        "conv.*"  # see example below: can freeze all (top-level) conv layers
        "^((?!classifier).)*$"   # freezes everything except for "classifier": useful for linear probing
        "conv1,layer1,layer2,layer3"  # freeze all layers up to layer3

    >>> model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1, 20, 5)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(20, 64, 5)),
          ('relu2', nn.ReLU())
        ]))
    >>> config["model.freeze_modules"]= "conv1"
    >>> _ = freeze_modules_based_on_opts(config, model)
    INFO    - Freezing module: conv1
    >>> model.train()
    >>> model.conv1.training
    False
    >>> model.conv2.training
    True
    """
    freeze_patterns = get_value_from_config(config, "model.freeze_modules", "")
    freeze_patterns = clean_strip(freeze_patterns)

    verbose = verbose 

    if freeze_patterns:
        # !! allow applying on just immediate chidren
        for name, module in model.named_children():
            if any([re.match(p, name) for p in freeze_patterns]):
                freeze_module(module)
                if verbose:
                    LOG.info("Freezing module: {}".format(name))

        for name, param in model.named_parameters(recurse=False):
            if any([re.match(p, name) for p in freeze_patterns]):
                param.requires_grad = False
                if verbose:
                    LOG.info("Freezing parameter: {}".format(name))

    if verbose and hasattr(model, "get_trainable_parameters"):
        param_list, _ = model.get_trainable_parameters()
        for params in param_list:
            if (
                not isinstance(params["param_names"], List)
                or not isinstance(params["params"], List)
                or not isinstance(params["weight_decay"], (float, int))
            ):
                param_types = {k: type(v) for k, v in params.items()}
                LOG.error(
                    "Expected parameter format: {{ params: List, weight_decay: float, param_names: List }}. "
                    "Got: {}".format(param_types)
                )
        # Flatten all parameter names
        trainable_param_names = [p for x in param_list for p in x["param_names"]]
        LOG.info("Trainable parameters: {}".format(trainable_param_names))

    return model


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


def unwrap_model_fn(model: torch.nn.Module) -> torch.nn.Module:
    """Helper function to unwrap the model.

    Args:
        model: An instance of torch.nn.Module.

    Returns:
        Unwrapped instance of torch.nn.Module.
    """
    unwrapped_model = model
    while True:
        if hasattr(unwrapped_model, "module"):
            # added by DataParallel and DistributedDataParallel
            unwrapped_model = unwrapped_model.module
        elif hasattr(unwrapped_model, "_fsdp_wrapped_module"):
            # added by FSDP
            unwrapped_model = unwrapped_model._fsdp_wrapped_module
        else:
            break
    return unwrapped_model




def check_frozen_norm_layer(model: torch.nn.Module) -> Tuple[bool, int]:
    from h264.src.models.modules.layers.normalization_layers import norm_layers_tuple

    unwrapped_model = unwrap_model_fn(model)

    count_norm = 0
    frozen_state = False
    for m in unwrapped_model.modules():
        if isinstance(m, norm_layers_tuple):
            frozen_state = m.weight.requires_grad

    return frozen_state, count_norm


def device_setup(config):
    """Helper function for setting up the device"""
    random_seed = get_value_from_config(config, "base.seed", 0)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)


    LOG.info("Random seeds are set to {}".format(random_seed))
    LOG.info("Using PyTorch version {}".format(torch.__version__))

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        LOG.warning("No GPUs available. Using CPU")
        device = torch.device("cpu")
        n_gpus = 0
    else:
        LOG.info("Available GPUs: {}".format(n_gpus))
        device = torch.device("cuda")

        if torch.backends.cudnn.is_available():
            import torch.backends.cudnn as cudnn

            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            LOG.info("CUDNN is enabled")

        allow_tf32 = not get_value_from_config(config, "base.disable_tf32", False)
        if torch.cuda.is_available():
            # TF32 is enabled by default in PyTorch < 1.12, but disabled in new versions.
            # See for details: https://github.com/pytorch/pytorch/issues/67384
            # Disable it using common.disable_tf32 flag
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32

    config["dev.device"]= device
    config["dev.num_gpus"]= n_gpus

    return config


def create_directories(dir_path: str, is_master_node: bool) -> None:
    """Helper function to create directories"""
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        LOG.info("Directory created at: {}".format(dir_path))
    else:
        LOG.info("Directory exists at: {}".format(dir_path))


def move_to_device(x: Any,
    device: Optional[str] = "cpu",
    non_blocking: Optional[bool] = True,
    *args,
    **kwargs
) -> Any:
    """Helper function to move data to a device"""
    if isinstance(x, Dict):
        for k, v in x.items():
            x[k] = move_to_device( x=v, device=device, non_blocking=non_blocking
            )

    elif isinstance(x, Tensor):
        # only tensors can be moved to a device
        x = x.to(device=device, non_blocking=non_blocking)
    elif isinstance(x, List):
        x = [move_to_device(a, device, non_blocking) for a in x]
    return x


